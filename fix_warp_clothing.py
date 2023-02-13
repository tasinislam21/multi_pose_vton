import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
import tqdm
import cv2
from models import networks
import os.path as osp
import torchvision.transforms as transforms
from PIL import Image
import os
from skimage import exposure

class Args:
    batchSize = 1
    datapairs = 'test_shuffle.txt'
    dataroot = 'data'
    phase = 'test'

opt = Args

def get_transform(normalize = True):
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class KeyDataset(data.Dataset):
    def __init__(self):
        super(KeyDataset, self).__init__()

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_candidateHD = os.path.join(opt.dataroot, opt.phase, 'candidateHD')

        self.dir_LabelHD = os.path.join(opt.dataroot, opt.phase, 'candidateHD_label')

        self.dir_denseHD = os.path.join(opt.dataroot, opt.phase, 'candidateHD_dense')

        self.dir_poseHD = osp.join(opt.dataroot, opt.phase, 'candidateHD_pose')

        self.dir_clothesHD = osp.join(opt.dataroot, opt.phase, 'clothesHD')
        self.dir_clothesMaskHD = osp.join(opt.dataroot, opt.phase, 'clothesHD_mask')

        self.init_categories(opt.dataroot ,opt.datapairs)
        self.transform = get_transform()
        self.transformBW = get_transform(normalize=False)

    def init_categories(self, dataroot, pairLst):
        self.human_names = []
        self.pose_names = []
        self.cloth_names = []
        with open(os.path.join(dataroot, pairLst), 'r') as f:
            for line in f.readlines():
                h_name, p_name, c_name = line.strip().split()
                self.human_names.append(h_name)
                self.pose_names.append(p_name)
                #self.cloth_names.append(c_name)
                self.cloth_names.append(h_name)

    def __getitem__(self, index):
        candidate_name = self.human_names[index]
        pose_name = self.pose_names[index]
        cloth_name = self.cloth_names[index]

        candidateHD_path = osp.join(self.dir_candidateHD, candidate_name)

        labelHD_path = osp.join(self.dir_LabelHD, candidate_name[:-4]+'.jpg.png')

        denseHD_path = osp.join(self.dir_denseHD, pose_name)
        source_dense_path = osp.join(self.dir_denseHD, candidate_name[:-4]+'_iuv.png')

        poseHD_path = osp.join(self.dir_poseHD, pose_name[:-8]+'_rendered.png')

        clothesHD_path = osp.join(self.dir_clothesHD, cloth_name)
        clothesHD_mask_path = osp.join(self.dir_clothesMaskHD, cloth_name)

        candidateHD_img = Image.open(candidateHD_path).convert('RGB')

        labelHD_img = Image.open(labelHD_path).convert('L')

        poseHD_img = Image.open(poseHD_path).convert('RGB')

        clothesHD_img = Image.open(clothesHD_path).convert('RGB')
        clothesHD_mask_img = Image.open(clothesHD_mask_path).convert('L')

        denseHD = np.array(Image.open(denseHD_path))
        denseHD_image = Image.open(denseHD_path).convert('RGB')
        source_denseHD = np.array(Image.open(source_dense_path))
        source_denseHD_image = Image.open(source_dense_path).convert('RGB')

        candidateHD = self.transform(candidateHD_img)
        labelHD = self.transformBW(labelHD_img) * 255
        poseHD = self.transform(poseHD_img)
        clothesHD = self.transform(clothesHD_img)
        denseHD = torch.from_numpy(denseHD).permute(2, 0, 1)
         #.permute(2, 0, 1)
        clothesHD_mask = self.transformBW(clothesHD_mask_img)
        denseHD_image = self.transform(denseHD_image)
        source_denseHD_image = self.transform(source_denseHD_image)

        return {'candidateHD': candidateHD, 'labelHD': labelHD, 'denseHD': denseHD, 'source_dense': source_denseHD,
                'clothesHD': clothesHD, 'clothesHD_mask': clothesHD_mask, 'poseHD': poseHD,
                'source_dense_image': source_denseHD_image,
                'dense_image': denseHD_image, 'name':candidate_name}

    def __len__(self):
            return len(self.human_names)

    def name(self):
        return 'KeyDataset'

t = KeyDataset()
t.initialize(opt)

dataloader = torch.utils.data.DataLoader(
            t,
            batch_size=opt.batchSize)

def generate_discrete_label(inputs, label_nc, onehot=True):
    pred_batch = []
    size = inputs.size()
    for input in inputs:
        input = input.view(1, label_nc, size[2], size[3])
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)
    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_map = []
    for p in pred_batch:
        p = p.view(1, 512, 512)
        label_map.append(p)
    label_map = torch.stack(label_map, 0)
    if not onehot:
        return label_map.float().cuda()
    size = label_map.size()
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
    return input_label

def ger_average_color(mask, arms):
    color = torch.zeros(arms.shape).cuda()
    for i in range(arms.shape[0]):
        count = len(torch.nonzero(mask[i, :, :, :]))
        if count < 10:
            color[i, 0, :, :] = 0
            color[i, 1, :, :] = 0
            color[i, 2, :, :] = 0
        else:
            color[i, 0, :, :] = arms[i, 0, :, :].sum() / count
            color[i, 1, :, :] = arms[i, 1, :, :].sum() / count
            color[i, 2, :, :] = arms[i, 2, :, :].sum() / count
    return color

sigmoid = nn.Sigmoid()
tanh = torch.nn.Tanh()

with torch.no_grad():
    G1n = networks.PHPM(input_nc=6, output_nc=4)
    G1n.cuda()
    G1n.load_state_dict(torch.load('checkpoint/phpm_45.pth'))
    G1n.eval()

with torch.no_grad():
    gmm = networks.GMM(input_nc=7, output_nc=3)
    gmm.cuda()
    gmm.load_state_dict(torch.load('checkpoint/gmm_free_disc_49.pth'))
    gmm.eval()

transform = get_transform()

def tensor2image(candidate, mask):
    candidate_numpy = (candidate[0].clone() + 1) * 0.5 * 255
    candidate_numpy = candidate_numpy.cpu().clamp(0, 255)
    candidate_numpy = candidate_numpy.detach().numpy().astype('uint8')
    candidate_numpy = candidate_numpy.swapaxes(0, 1).swapaxes(1, 2)

    mask_numpy = (mask[0].clone() + 1) * 0.5 * 255
    mask_numpy = mask_numpy.cpu().clamp(0, 255)
    mask_numpy = mask_numpy.detach().numpy().astype('uint8')
    mask_numpy = mask_numpy.swapaxes(0, 1).swapaxes(1, 2)
    mask_numpy = np.where(mask_numpy == 127, 0, mask_numpy)

    candidate_numpy = cv2.bitwise_and(candidate_numpy, candidate_numpy, mask=mask_numpy)

    return candidate_numpy

step = 0


for data in tqdm.tqdm(dataloader):
    name = data['name']
    in_garmentHD = data['clothesHD'].cuda()
    in_clothesHD_mask = data['clothesHD_mask'].cuda()
    pre_clothes_mask = (in_clothesHD_mask > 0.5).float().cuda()
    in_garmentHD *= pre_clothes_mask
    in_denseHD = data['denseHD'].cuda()
    in_skeletonHD = data['poseHD'].cuda()
    in_garmentHD = torch.nn.functional.pad(input=in_garmentHD, pad=(82, 82, 0, 0), mode='constant', value=0)
    in_denseHD = torch.nn.functional.pad(input=in_denseHD, pad=(82, 82, 0, 0), mode='constant', value=0)
    in_skeletonHD = torch.nn.functional.pad(input=in_skeletonHD, pad=(82, 82, 0, 0), mode='constant', value=0)
    pre_clothes_mask = torch.nn.functional.pad(input=pre_clothes_mask, pad=(82, 82, 0, 0), mode='constant', value=0)

    G1_in = torch.cat([in_garmentHD, in_denseHD], dim=1)
    arm_label = G1n(G1_in)
    arm_label = sigmoid(arm_label)
    arm_label_discrete = generate_discrete_label(arm_label, 4, False)

    target_mask_clothes = (arm_label_discrete == 1).float()
    warped_garment, affine = gmm(in_garmentHD, target_mask_clothes, in_skeletonHD)
    warped_garment *= target_mask_clothes
    warped_garment = tanh(warped_garment)

    normal_clothing = tensor2image(in_garmentHD[:, :, :, int(82):512 - int(82)], pre_clothes_mask[:, :, :, int(82):512 - int(82)])
    img = Image.fromarray(normal_clothing)
    img.save('garment_result/' + name[0][:-4] + '_original_clothing.jpg')

    warped = tensor2image(warped_garment[:, :, :, int(82):512 - int(82)], target_mask_clothes[:, :, :, int(82):512 - int(82)])
    img = Image.fromarray(warped)
    img.save('garment_result/' + name[0][:-4]+'_warped.jpg')

    #histogram_matched = fixHistogram(warped_garment[:, :, :, int(82):512 - int(82)], in_garmentHD[:, :, :, int(82):512 - int(82)], target_mask_clothes[:, :, :, int(82):512 - int(82)])
    #img = Image.fromarray(histogram_matched)
    #img.save('garment_result/' + name[0][:-4] + '_histogram_matched.jpg')

    step += 1