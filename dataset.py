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