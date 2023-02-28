from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
import torch
import random
import pickle

def get_transform(normalize=True):
    transform_list = []
    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class DeepFashionDataset(Dataset):
    def __init__(self, path, phase, size):
        self.phase = phase  # train or test
        self.size = size    # 256 or 512  FOR  174x256 or 348x512

        # set root directories
        self.image_root = os.path.join(path, 'DeepFashion_highres', phase)
        self.densepose_root = os.path.join(path, 'densepose', phase)
        self.parsing_root = os.path.join(path, 'silhouette', phase)
        # path to pairs of data
        pairs_csv_path = os.path.join(path, 'DeepFashion_highres', 'tools', 'fashion-pairs-%s.csv'%phase)

        # uv space
        self.uv_root = os.path.join(path, 'complete_coordinates', phase)

        # initialize the pairs of data
        self.init_pairs(pairs_csv_path)
        self.data_size = len(self.pairs)
        print('%s data pairs (#=%d)...'%(phase, self.data_size))

        if phase == 'train':
            # get dictionary of image name and transfrom to detect and align the face
            with open(os.path.join(path, 'resources', 'train_face_T.pickle'), 'rb') as handle:
                self.faceTransform = pickle.load(handle)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def init_pairs(self, pairs_csv_path):
        pairs_file = pd.read_csv(pairs_csv_path)
        self.pairs = []
        self.sources = {}
        print('Loading data pairs ...')
        for i in range(len(pairs_file)):
            pair = [pairs_file.iloc[i]['from'], pairs_file.iloc[i]['to']]
            self.pairs.append(pair)
        print('Loading data pairs finished ...')


    def __len__(self):
        return self.data_size


    def resize_height_PIL(self, x, height=512):
        w, h = x.size
        width  = int(height * w / h)
        return x.resize((width, height), Image.NEAREST) #Image.ANTIALIAS


    def resize_PIL(self, x, height=512, width=348, type=Image.NEAREST):
        return x.resize((width, height), type)


    def tensors2square(self, im, pose, sil):
        width = im.shape[2]
        diff = self.size - width
        if self.phase == 'train':
            left = random.randint(0, diff)
            right = diff - left
        else: # when testing put in the center
            left = int((self.size-width)/2)
            right = diff - left
        im = torch.nn.functional.pad(input=im, pad=(right, left, 0, 0), mode='constant', value=0)
        pose = torch.nn.functional.pad(input=pose, pad=(right, left, 0, 0), mode='constant', value=0)
        sil = torch.nn.functional.pad(input=sil, pad=(right, left, 0, 0), mode='constant', value=0)
        return im, pose, sil, left, right


    def __getitem__(self, index):
        # get current pair
        im1_name, im2_name = self.pairs[index]

        # get path to dataset
        input_image_path = os.path.join(self.image_root, im1_name)
        target_image_path = os.path.join(self.image_root, im2_name)
        # dense pose
        input_densepose_path = os.path.join(self.densepose_root, im1_name.split('.')[0]+'_iuv.png')
        target_densepose_path = os.path.join(self.densepose_root, im2_name.split('.')[0]+'_iuv.png')
        # silhouette
        input_sil_path = os.path.join(self.parsing_root, im1_name.split('.')[0]+'_sil.png')
        target_sil_path = os.path.join(self.parsing_root, im2_name.split('.')[0]+'_sil.png')
        # uv space
        complete_coor_path = os.path.join(self.uv_root, im1_name.split('.')[0]+'_uv_coor.npy')

        # read data
        # get original size of data -> for augmentation
        input_image_pil = Image.open(input_image_path).convert('RGB')
        orig_w, orig_h = input_image_pil.size
        if self.phase == 'test':
            # set target height and target width
            if self.size == 512:
                target_h = 512
                target_w = 348
            if self.size == 256:
                target_h = 256
                target_w = 174
            # images
            input_image = self.resize_PIL(input_image_pil, height=target_h, width=target_w, type=Image.ANTIALIAS)
            target_image = self.resize_PIL(Image.open(target_image_path).convert('RGB'), height=target_h, width=target_w, type=Image.ANTIALIAS)
            # dense pose
            input_densepose = np.array(self.resize_PIL(Image.open(input_densepose_path), height=target_h, width=target_w))
            target_densepose = np.array(self.resize_PIL(Image.open(target_densepose_path), height=target_h, width=target_w))
            # silhouette
            silhouette1 = np.array(self.resize_PIL(Image.open(input_sil_path), height=target_h, width=target_w))/255
            silhouette2 = np.array(self.resize_PIL(Image.open(target_sil_path), height=target_h, width=target_w))/255
            # union with densepose mask for a more accurate mask
            silhouette1 = 1-((1-silhouette1) * (input_densepose[:, :, 0] == 0).astype('float'))

        else:
            input_image = self.resize_height_PIL(input_image_pil, self.size)
            target_image = self.resize_height_PIL(Image.open(target_image_path).convert('RGB'), self.size)
            # dense pose
            input_densepose = np.array(self.resize_height_PIL(Image.open(input_densepose_path), self.size))
            target_densepose = np.array(self.resize_height_PIL(Image.open(target_densepose_path), self.size))
            # silhouette
            silhouette1 = np.array(self.resize_height_PIL(Image.open(input_sil_path), self.size))/255
            silhouette2 = np.array(self.resize_height_PIL(Image.open(target_sil_path), self.size))/255
            # union with densepose masks
            silhouette1 = 1-((1-silhouette1) * (input_densepose[:, :, 0] == 0).astype('float'))
            silhouette2 = 1-((1-silhouette2) * (target_densepose[:, :, 0] == 0).astype('float'))

        # read uv-space data
        complete_coor = np.load(complete_coor_path)

        # Transform
        input_image = self.transform(input_image)
        target_image = self.transform(target_image)
        # Dense Pose
        input_densepose = torch.from_numpy(input_densepose).permute(2, 0, 1)
        target_densepose = torch.from_numpy(target_densepose).permute(2, 0, 1)
        # silhouette
        silhouette1 = torch.from_numpy(silhouette1).float().unsqueeze(0) # from h,w to c,h,w
        silhouette2 = torch.from_numpy(silhouette2).float().unsqueeze(0) # from h,w to c,h,w

        # put into a square
        input_image, input_densepose, silhouette1, Sleft, Sright = self.tensors2square(input_image, input_densepose, silhouette1)
        target_image, target_densepose, silhouette2, Tleft, Tright = self.tensors2square(target_image, target_densepose, silhouette2)

        if self.phase == 'train':
            # remove loaded center shift and add augmentation shift
            loaded_shift = int((orig_h-orig_w)/2)
            complete_coor = ((complete_coor+1)/2)*(orig_h-1) # [-1, 1] to [0, orig_h]
            complete_coor[:,:,0] = complete_coor[:,:,0] - loaded_shift # remove center shift
            complete_coor = ((2*complete_coor/(orig_h-1))-1) # [0, orig_h] (no shift in w) to [-1, 1]
            complete_coor = ((complete_coor+1)/2) * (self.size-1) # [-1, 1] to [0, size] (no shift in w)
            complete_coor[:,:,0] = complete_coor[:,:,0] + Sright # add augmentation shift to w
            complete_coor = ((2*complete_coor/(self.size-1))-1) # [0, size] (with shift in w) to [-1,1]
            # to tensor
            complete_coor = torch.from_numpy(complete_coor).float().permute(2, 0, 1)
        else:
            # might have hxw inconsistencies since dp is of different sizes.. fixing this..
            loaded_shift = int((orig_h-orig_w)/2)
            complete_coor = ((complete_coor+1)/2)*(orig_h-1) # [-1, 1] to [0, orig_h]
            complete_coor[:,:,0] = complete_coor[:,:,0] - loaded_shift # remove center shift
            # before: width complete_coor[:,:,0] 0-orig_w-1
            # and    height complete_coor[:,:,1] 0-orig_h-1
            complete_coor[:,:,0] = (complete_coor[:,:,0]/(orig_w-1))*(target_w-1)
            complete_coor[:,:,1] = (complete_coor[:,:,1]/(orig_h-1))*(target_h-1)
            complete_coor[:,:,0] = complete_coor[:,:,0] + Sright # add center shift to w
            complete_coor = ((2*complete_coor/(self.size-1))-1) # [0, size] (with shift in w) to [-1,1]
            # to tensor
            complete_coor = torch.from_numpy(complete_coor).float().permute(2, 0, 1)

        # either source or target pass 1:5
        if self.phase == 'train':
            choice = random.randint(0, 6)
            if choice == 0:
                # source pass
                target_im = input_image
                target_p = input_densepose
                target_sil = silhouette1
                target_image_name = im1_name
                target_left_pad = Sleft
                target_right_pad = Sright
            else:
                # target pass
                target_im = target_image
                target_p = target_densepose
                target_sil = silhouette2
                target_image_name = im2_name
                target_left_pad = Tleft
                target_right_pad = Tright
        else:
            target_im = target_image
            target_p = target_densepose
            target_sil = silhouette2
            target_image_name = im2_name
            target_left_pad = Tleft
            target_right_pad = Tright

        # Get the face transfrom
        if self.phase == 'train':
            if target_image_name in self.faceTransform.keys():
                FT = torch.from_numpy(self.faceTransform[target_image_name]).float()
            else:   # no face detected
                FT = torch.zeros((3,3))

        # return data
        if self.phase == 'train':
            return {'input_image':input_image, 'target_image':target_im,
                    'target_sil': target_sil,
                    'target_pose':target_p,
                    'TargetFaceTransform': FT, 'target_left_pad':torch.tensor(target_left_pad), 'target_right_pad':torch.tensor(target_right_pad),
                    'input_sil': silhouette1,  'complete_coor':complete_coor,
                    }

        if self.phase == 'test':
            save_name = im1_name.split('.')[0] + '_2_' + im2_name.split('.')[0] + '_vis.png'
            return {'input_image':input_image, 'target_image':target_im,
                    'target_sil': target_sil,
                    'target_pose':target_p,
                    'target_left_pad':torch.tensor(target_left_pad), 'target_right_pad':torch.tensor(target_right_pad),
                    'input_sil': silhouette1,  'complete_coor':complete_coor,
                    'save_name':save_name,
                    }

class CustomDeepFashionDataset(Dataset):
    def initialize(self, opt, size):
        self.opt = opt
        self.size = size
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.phase + '_pt') #person images
        self.dir_Label = os.path.join(opt.dataroot, opt.phase + '_label_pt')
        self.dir_skeleton = os.path.join(opt.dataroot, opt.phase + '_rgbSkeleton_pt')
        self.dir_torso = os.path.join(opt.dataroot, opt.phase + '_torso_pt')
        self.dir_arm1 = os.path.join(opt.dataroot, opt.phase + '_arm1_pt')
        self.dir_arm2 = os.path.join(opt.dataroot, opt.phase + '_arm2_pt')
        self.dir_skin = os.path.join(opt.dataroot, opt.phase + '_skin_pt')
        self.dir_uv = os.path.join(opt.dataroot, opt.phase + '_complete_coordinates')
        self.dir_dense = os.path.join(opt.dataroot, opt.phase + '_dense')

        self.init_categories(opt.pairLst)
        with open(os.path.join(opt.dataroot, 'resources', 'saved_faces.pkl'), 'rb') as handle:
            self.faceBoxes = pickle.load(handle)

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        self.dataSize = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.dataSize):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)
        print('Loading data pairs finished ...')

    def shiftValue(self):
        width = 192
        diff = self.size - width
        left = random.randint(0, diff)
        right = diff - left
        return left, right

    def __getitem__(self, index):
        P1_name, P2_name = self.pairs[index]
        P1_path = os.path.join(self.dir_P, P1_name[:-3]+'pt') # person 1
        P2_path = os.path.join(self.dir_P, P2_name[:-3]+'pt')  # person 2
        Label_path = os.path.join(self.dir_Label, P1_name[:-3]+'pt')
        Label2_path = os.path.join(self.dir_Label, P2_name[:-3]+'pt')
        Torso_path = os.path.join(self.dir_torso, P1_name[:-3] + 'pt')
        Arm1_path = os.path.join(self.dir_arm1, P1_name[:-3] + 'pt')
        Arm2_path = os.path.join(self.dir_arm2, P1_name[:-3] + 'pt')
        complete_coor_path = os.path.join(self.dir_uv, P1_name.split('.')[0] + '_uv_coor.npy')
        dense_path = os.path.join(self.dir_dense, P2_name.split('.')[0] + '_iuv.png')
        transform_B = get_transform()

        Sleft, Sright = self.shiftValue()
        Tleft, Tright = self.shiftValue()

        orig_h = 256
        orig_w = 192
        choice = random.randint(0, 6)
        if choice != 0:
            P1 = torch.load(P1_path)
            P1 = torch.nn.functional.pad(input=P1, pad=(Sright, Sleft, 0, 0), mode='constant', value=0)
            P2 = torch.load(P2_path)
            P2 = torch.nn.functional.pad(input=P2, pad=(Tright, Tleft, 0, 0), mode='constant', value=0)
            P2_pathHD = os.path.join('custom_DeepFashionV2/train', P2_name)
            P2HD = Image.open(P2_pathHD).convert('RGB')
            P2HD = transform_B(P2HD)
            Label = torch.load(Label_path)
            Label = torch.nn.functional.pad(input=Label, pad=(Sright, Sleft, 0, 0), mode='constant', value=0)
            Label2 = torch.load(Label2_path)
            Label2 = torch.nn.functional.pad(input=Label2, pad=(Tright, Tleft, 0, 0), mode='constant', value=0)
            #Skeleton = torch.load(Skeleton_path)
            Torso = torch.load(Torso_path)
            Torso = torch.nn.functional.pad(input=Torso, pad=(Sright, Sleft, 0, 0), mode='constant', value=0)
            Arm1 = torch.load(Arm1_path)
            Arm1 = torch.nn.functional.pad(input=Arm1, pad=(Sright, Sleft, 0, 0), mode='constant', value=0)
            Arm2 = torch.load(Arm2_path)
            Arm2 = torch.nn.functional.pad(input=Arm2, pad=(Sright, Sleft, 0, 0), mode='constant', value=0)
            #Skin = torch.load(Skin_path)
            Boxes = torch.from_numpy(self.faceBoxes[P2_name[:-4]][0]).int()
            targetDense = np.array(Image.open(dense_path))
            targetDense = torch.from_numpy(targetDense).permute(2, 0, 1)
            targetDense = torch.nn.functional.pad(input=targetDense, pad=(Tright, Tleft, 0, 0), mode='constant', value=0)

            complete_coor = np.load(complete_coor_path)
            loaded_shift = int((orig_h - orig_w) / 2)
            complete_coor = ((complete_coor + 1) / 2) * (orig_h - 1)  # [-1, 1] to [0, orig_h]
            complete_coor[:, :, 0] = complete_coor[:, :, 0] - loaded_shift  # remove center shift
            complete_coor = ((2 * complete_coor / (orig_h - 1)) - 1)  # [0, orig_h] (no shift in w) to [-1, 1]
            complete_coor = ((complete_coor + 1) / 2) * (self.size - 1)  # [-1, 1] to [0, size] (no shift in w)
            complete_coor[:, :, 0] = complete_coor[:, :, 0] + Sright  # add augmentation shift to w
            complete_coor = ((2 * complete_coor / (self.size - 1)) - 1)  # [0, size] (with shift in w) to [-1,1]
            # to tensor
            complete_coor = torch.from_numpy(complete_coor).float().permute(2, 0, 1)

            return {'P1': P1, 'P2': P2, 'Label': Label, #'Skeleton': Skeleton,
                    'torso': Torso, 'arm1': Arm1, 'arm2': Arm2,
                    'Label2': Label2, 'P2HD': P2HD,
                    #'Skin': Skin,
                    'Boxes': Boxes, 'complete_coor': complete_coor,
                    'target_dense': targetDense, 'Tleft': Tleft,
                    'Tright': Tright}
        else:
            P1 = torch.load(P2_path)
            P1 = torch.nn.functional.pad(input=P1, pad=(Sright, Sleft, 0, 0), mode='constant', value=0)
            P2 = torch.load(P1_path)
            P2 = torch.nn.functional.pad(input=P2, pad=(Tright, Tleft, 0, 0), mode='constant', value=0)
            P2_pathHD = os.path.join('custom_DeepFashionV2/train', P1_name)
            P2HD = Image.open(P2_pathHD).convert('RGB')
            P2HD = transform_B(P2HD)
            Label = torch.load(Label2_path)
            Label = torch.nn.functional.pad(input=Label, pad=(Sright, Sleft, 0, 0), mode='constant', value=0)
            Label2 = torch.load(Label_path)
            Label2 = torch.nn.functional.pad(input=Label2, pad=(Tright, Tleft, 0, 0), mode='constant', value=0)
            # Skeleton = torch.load(Skeleton_path)
            Torso_path = os.path.join(self.dir_torso, P2_name[:-3] + 'pt')
            Arm1_path = os.path.join(self.dir_arm1, P2_name[:-3] + 'pt')
            Arm2_path = os.path.join(self.dir_arm2, P2_name[:-3] + 'pt')
            complete_coor_path = os.path.join(self.dir_uv, P2_name.split('.')[0] + '_uv_coor.npy')
            dense_path = os.path.join(self.dir_dense, P1_name.split('.')[0] + '_iuv.png')
            Torso = torch.load(Torso_path)
            Torso = torch.nn.functional.pad(input=Torso, pad=(Sright, Sleft, 0, 0), mode='constant', value=0)
            Arm1 = torch.load(Arm1_path)
            Arm1 = torch.nn.functional.pad(input=Arm1, pad=(Sright, Sleft, 0, 0), mode='constant', value=0)
            Arm2 = torch.load(Arm2_path)
            Arm2 = torch.nn.functional.pad(input=Arm2, pad=(Sright, Sleft, 0, 0), mode='constant', value=0)
            # Skin = torch.load(Skin_path)
            Boxes = torch.from_numpy(self.faceBoxes[P2_name[:-4]][0]).int()
            targetDense = np.array(Image.open(dense_path))
            targetDense = torch.from_numpy(targetDense).permute(2, 0, 1)
            targetDense = torch.nn.functional.pad(input=targetDense, pad=(Tright, Tleft, 0, 0), mode='constant',
                                                  value=0)

            complete_coor = np.load(complete_coor_path)
            loaded_shift = int((orig_h - orig_w) / 2)
            complete_coor = ((complete_coor + 1) / 2) * (orig_h - 1)  # [-1, 1] to [0, orig_h]
            complete_coor[:, :, 0] = complete_coor[:, :, 0] - loaded_shift  # remove center shift
            complete_coor = ((2 * complete_coor / (orig_h - 1)) - 1)  # [0, orig_h] (no shift in w) to [-1, 1]
            complete_coor = ((complete_coor + 1) / 2) * (self.size - 1)  # [-1, 1] to [0, size] (no shift in w)
            complete_coor[:, :, 0] = complete_coor[:, :, 0] + Sright  # add augmentation shift to w
            complete_coor = ((2 * complete_coor / (self.size - 1)) - 1)  # [0, size] (with shift in w) to [-1,1]
            # to tensor
            complete_coor = torch.from_numpy(complete_coor).float().permute(2, 0, 1)

            return {'P1': P1, 'P2': P2, 'Label': Label,  # 'Skeleton': Skeleton,
                    'torso': Torso, 'arm1': Arm1, 'arm2': Arm2,
                    'Label2': Label2, 'P2HD': P2HD,
                    # 'Skin': Skin,
                    'Boxes': Boxes, 'complete_coor': complete_coor,
                    'target_dense': targetDense, 'Tleft': Tleft,
                    'Tright': Tright}

    def __len__(self):
            return self.dataSize

    def name(self):
        return 'KeyDataset'

class CustomDeepFashionDatasetHD(Dataset):
    def initialize(self, opt, size):
        self.opt = opt
        self.size = size
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.phase)
        self.dir_HD = os.path.join(opt.dataroot, opt.phase +'_HD')
        self.dir_Label = os.path.join(opt.dataroot, opt.phase + '_label')
        self.dir_uv = os.path.join(opt.dataroot, opt.phase + '_complete_coordinates')
        self.dir_dense = os.path.join(opt.dataroot, opt.phase + '_dense')

        self.init_categories(opt.pairLst)
        with open(os.path.join(opt.dataroot, 'resources', 'train_face_T.pickle'), 'rb') as handle:
            self.faceTransform = pickle.load(handle)

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        self.dataSize = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.dataSize):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)
        print('Loading data pairs finished ...')

    def shiftValue(self):
        width = 348
        diff = self.size - width
        left = random.randint(0, diff)
        right = diff - left
        return left, right

    def __getitem__(self, index):
        choice = random.randint(0, 6)
        if choice != 0:
            P1_name, P2_name = self.pairs[index]
        else:
            P2_name, P1_name = self.pairs[index]

        P1_path = os.path.join(self.dir_P, P1_name)  # person 1
        P2_path = os.path.join(self.dir_P, P2_name)  # person 2
        P2_pathHD = os.path.join(self.dir_HD, P2_name)
        Label_path = os.path.join(self.dir_Label, P1_name + '.png')
        Label2_path = os.path.join(self.dir_Label, P2_name + '.png')
        dense_path = os.path.join(self.dir_dense, P2_name.split('.')[0] + '_iuv.png')
        complete_coor_path = os.path.join(self.dir_uv, P1_name.split('.')[0] + '_iuv_uv_coor.npy')
        transform_A = get_transform(normalize=False)
        transform_B = get_transform()

        if P2_name in self.faceTransform.keys():
            Boxes = torch.from_numpy(self.faceTransform[P2_name]).float()
        else:
            Boxes = torch.zeros((3, 3))

        Sleft, Sright = self.shiftValue()
        Tleft, Tright = self.shiftValue()

        orig_h = 512
        orig_w = 348

        P1 = Image.open(P1_path).convert('RGB')
        P1 = transform_B(P1)
        P1 = torch.nn.functional.pad(input=P1, pad=(Sright, Sleft, 0, 0), mode='constant', value=0)
        P2 = Image.open(P2_path).convert('RGB')
        P2 = transform_B(P2)
        P2 = torch.nn.functional.pad(input=P2, pad=(Tright, Tleft, 0, 0), mode='constant', value=0)
        P2HD = Image.open(P2_pathHD).convert('RGB')
        P2HD = transform_B(P2HD)
        Label = Image.open(Label_path).convert('L')
        Label = transform_A(Label) * 255
        Label = torch.nn.functional.pad(input=Label, pad=(Sright, Sleft, 0, 0), mode='constant', value=0)
        Label2 = Image.open(Label2_path).convert('L')
        Label2 = transform_A(Label2) * 255
        Label2 = torch.nn.functional.pad(input=Label2, pad=(Tright, Tleft, 0, 0), mode='constant', value=0)

        targetDense = np.array(Image.open(dense_path))
        targetDense = torch.from_numpy(targetDense).permute(2, 0, 1)
        targetDense = torch.nn.functional.pad(input=targetDense, pad=(Tright, Tleft, 0, 0), mode='constant', value=0)

        complete_coor = np.load(complete_coor_path)
        loaded_shift = int((orig_h - orig_w) / 2)
        complete_coor = ((complete_coor + 1) / 2) * (orig_h - 1)  # [-1, 1] to [0, orig_h]
        complete_coor[:, :, 0] = complete_coor[:, :, 0] - loaded_shift  # remove center shift
        complete_coor = ((2 * complete_coor / (orig_h - 1)) - 1)  # [0, orig_h] (no shift in w) to [-1, 1]
        complete_coor = ((complete_coor + 1) / 2) * (self.size - 1)  # [-1, 1] to [0, size] (no shift in w)
        complete_coor[:, :, 0] = complete_coor[:, :, 0] + Sright  # add augmentation shift to w
        complete_coor = ((2 * complete_coor / (self.size - 1)) - 1)  # [0, size] (with shift in w) to [-1,1]
        # to tensor
        complete_coor = torch.from_numpy(complete_coor).float().permute(2, 0, 1)

        return {'P1': P1, 'P2': P2, 'P2HD': P2HD, 'Label': Label, 'Label2': Label2,
                'Boxes': Boxes, 'complete_coor': complete_coor,
                'target_dense': targetDense, 'Tleft': Tleft,
                'Tright': Tright, 'name':P1_name}

    def __len__(self):
            return self.dataSize

    def name(self):
        return 'KeyDataset'
