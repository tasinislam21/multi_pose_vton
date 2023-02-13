#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
import cv2
from torch.autograd import Variable
from models import networks
import os.path as osp
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from face_detect import extract_face


# In[2]:


class Args:
    batchSize = 1
    #dataroot = '../../DeepFashion_Try_On/acgpn_dataset'
    datapairs = 'custom_test.csv'
    dataroot = 'data'
    #datapairs = 'test_pairs.txt'
    phase = 'test'

opt = Args


# In[3]:


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
        self.dir_candidate = osp.join(opt.dataroot, opt.phase, opt.phase + '_img')
        self.dir_pose = osp.join(opt.dataroot, opt.phase, opt.phase + '_posergb')
        self.dir_label = osp.join(opt.dataroot, opt.phase, opt.phase + '_label')
        self.dir_clothes = osp.join(opt.dataroot, opt.phase, opt.phase + '_color')
        self.dir_edge = osp.join(opt.dataroot, opt.phase, opt.phase + '_edge')
        self.dir_face = osp.join(opt.dataroot, opt.phase, opt.phase + '_face')

        self.init_categories(opt.dataroot ,opt.datapairs)
        self.transform = get_transform()
        self.transformBW = get_transform(normalize=False)

    def init_categories(self, dataroot, pairLst):
        file = pd.read_csv(osp.join(dataroot,pairLst))
        self.size = len(file)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            self.pairs.append(file.iloc[i])

        print('Loading data pairs finished ...')

    def __getitem__(self, index):
        candidate_name = self.pairs[index]['from']
        clothes_name = self.pairs[index]['clothes']
        original_pose_name = self.pairs[index]['pose1']

        candidate_path = osp.join(self.dir_candidate, candidate_name)
        label_path = osp.join(self.dir_label, candidate_name[:-3]+'png')
        BP1_path = osp.join(self.dir_pose, original_pose_name)
        clothes_path = osp.join(self.dir_clothes, clothes_name)
        edge_path = osp.join(self.dir_edge, clothes_name)
        face_path = osp.join(self.dir_face, candidate_name)
        #imgmask_path = osp.join(self.dir_imgmask, candidate_name[:-3]+'png')

        P1_img = Image.open(candidate_path).convert('RGB')
        P1_bw_img = Image.open(candidate_path).convert('L')
        BP1_img = Image.open(BP1_path).convert('RGB')
        Face_img = Image.open(face_path).convert('RGB')
        label_img = Image.open(label_path).convert('L')

        clothes_img = Image.open(clothes_path)
        edge_img = Image.open(edge_path).convert('L')

        candidate = self.transform(P1_img)
        candidate_bw = self.transformBW(P1_bw_img)
        pose = self.transform(BP1_img)
        label = self.transformBW(label_img) * 255
        clothes = self.transform(clothes_img)
        edge = self.transformBW(edge_img)
        face = self.transform(Face_img)

        return {'candidate': candidate, 'candidateBW': candidate_bw, 'pose1': pose,
                'P1_path': candidate_name, 'label': label, 'clothes': clothes,
                'edge': edge, 'face': face}

    def __len__(self):
            return self.size

    def name(self):
        return 'KeyDataset'


# In[4]:


pose = pd.read_csv(osp.join(opt.dataroot,opt.phase,"pose.csv"), header=None)


# In[5]:


t = KeyDataset()
t.initialize(opt)

dataloader = torch.utils.data.DataLoader(
            t,
            batch_size=opt.batchSize)


# In[6]:


len(dataloader)


# In[7]:


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

def morpho(mask, iter, bigger=True):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    new = []
    for i in range(len(mask)):
        tem = mask[i].cpu().detach().numpy().squeeze().reshape(256, 192, 1)*255
        tem = tem.astype(np.uint8)
        if bigger:
            tem = cv2.dilate(tem, kernel, iterations=iter)
        else:
            tem = cv2.erode(tem, kernel, iterations=iter)
        tem = tem.astype(np.float64)
        tem = tem.reshape(1, 256, 192)
        new.append(tem.astype(np.float64)/255.0)
    new = np.stack(new)
    new = torch.FloatTensor(new).cuda()
    return new

def encode(label_map, size):
    label_nc = 14
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
    return input_label

def grid_sample(input, grid, canvas=None):
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output

def get_average_color(mask, arms):
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


# In[8]:


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
        p = p.view(1, 256, 192)
        label_map.append(p)
    label_map = torch.stack(label_map, 0)
    if not onehot:
        return label_map.float().cuda()
    size = label_map.size()
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
    return input_label

sigmoid = nn.Sigmoid()
tanh = torch.nn.Tanh()


# In[9]:


with torch.no_grad():
    G1n = networks.PHPM(input_nc=6, output_nc=4)
    G1n.cuda()
    G1n.load_state_dict(torch.load('checkpoint/phpm_49.pth'))
    G1n.eval()

with torch.no_grad():
    gmm = networks.GMM(input_nc=7, output_nc=3)
    gmm.cuda()
    gmm.load_state_dict(torch.load('checkpoint/GMM.pth'))
    gmm.eval()

with torch.no_grad():
    G3 = networks.Unet(input_nc=14)
    G3.cuda()
    G3.load_state_dict(torch.load('checkpoint/unet_final_full_body.pth'))
    G3.eval()

with torch.no_grad():
    DeepFake = networks.AutoEncoder()
    DeepFake.cuda()
    DeepFake.load_state_dict(torch.load('checkpoint/deepFake.pt'))
    DeepFake.eval()


# In[10]:


transform = get_transform()

def tensor2image(tensor):
    tensor = (tensor[0].clone() + 1) * 0.5 * 255
    tensor = tensor.cpu().clamp(0, 255)
    array = tensor.detach().numpy().astype('uint8')
    array = array.swapaxes(0, 1).swapaxes(1, 2)
    return array

def tensor2fake(tensor):
    tensor = (tensor.clone() + 1) * 0.5 * 255
    tensor = tensor.cpu().clamp(0, 255)
    array = tensor.detach().numpy().astype('uint8')
    array = array.swapaxes(1, 2).swapaxes(2, 3)
    return array


# In[11]:


position = -1
for data in dataloader:
    position +=1
    if position == 3:
        in_label = data['label'].cuda()
        mask_background = (in_label == 0).float()
        mask_clothes = (in_label == 4).float()
        mask_hair = (in_label == 1).float()
        mask_bottom = (in_label == 8).float()
        mask_head = (in_label == 12).float()
        mask_arm1 = (in_label == 11).float()
        mask_arm2 = (in_label == 13).float()

        in_P1 = data['candidate'].cuda()
        mask_fore = (in_label > 0).float()
        img_fore = in_P1 * mask_fore
        in_edge = Variable(data['edge'].cuda())
        in_img_fore = Variable(img_fore.cuda())
        in_color = data['clothes'].cuda()
        in_image = Variable(data['candidate'].cuda())
        in_mask_fore = Variable(mask_fore.cuda())
        in_skeleton = Variable(data['pose1'].cuda())
        pre_clothes_mask = (in_edge > 0.5).long()
        clothes = in_color*pre_clothes_mask
        size = in_label.size()

        arm1_mask = morpho(mask_arm1, 2, True)
        arm2_mask = morpho(mask_arm2, 2, True)
        torso_label_large = morpho(mask_clothes, 2, True)

        img_hole_hand = img_fore * (1 - torso_label_large) * (1 - arm1_mask) * (1 - arm2_mask)
        skin = get_average_color((mask_arm1 + mask_arm2 - mask_arm2 * mask_arm1),
                                   (mask_arm1 + mask_arm2 - mask_arm2 * mask_arm1) * data['candidate'].cuda())
        break


# In[ ]:


sigmoid = nn.Sigmoid()

tanh = nn.Tanh()
for i in range(len(pose)):
    pose_name = pose.iloc[i].tolist()[0]
    pose1 = transform(Image.open("data/test/test_posergb/"+pose_name).convert('RGB')).unsqueeze(0).cuda()
    G1_in = torch.cat([in_color, pose1], dim=1)
    arm_label = G1n(G1_in)
    #arm_label = G1n(in_color, pose1)
    arm_label = sigmoid(arm_label)
    arm_label_discrete = generate_discrete_label(arm_label, 4, False)
    ground_truth_map_torso = (arm_label_discrete == 1).float()
    fake_c, warped = gmm(clothes, ground_truth_map_torso, pose1)
    fake_c = tanh(fake_c)

    warped_garment = fake_c
    shape_data = torch.cat([img_hole_hand, warped_garment], 1)
    target_mask_arm1 = (arm_label_discrete == 2).float()
    target_mask_arm2 = (arm_label_discrete == 3).float()
    arms = torch.cat([target_mask_arm1, target_mask_arm2], 1)
    g3_input = torch.cat([img_hole_hand, warped_garment, arms, skin, pose1], 1)
    full_body = G3(g3_input)
    full_body = tanh(full_body)

    full_body = tensor2fake(full_body)

    generated_face = extract_face(full_body[0])
    generated_face = cv2.resize(generated_face, (128, 128))
    frame = np.array(full_body)
    generated_face = cv2.cvtColor(generated_face, cv2.COLOR_BGR2RGB)
    img_tensor = generated_face[:, :, ::-1].transpose((2, 0, 1)).copy()  # chw, RGB order,[0,255]
    img_tensor = torch.from_numpy(img_tensor).float().div(255)  # chw , FloatTensor type,[0,1]
    img_tensor = img_tensor.unsqueeze(0)  # nch*w
    x = img_tensor.to('cuda')
    out = DeepFake(x, version='b')
    out = tensor2image(out)
    break
    #pose1 = Image.fromarray(pose1)
    #pose1.save("data/result/beta/"+str(i)+".jpg")


# In[31]:


Image.fromarray(full_body[0])


# In[ ]:


plt.imshow(arm_label_discrete[0].permute(1,2,0).detach().cpu().numpy(), cmap='gray')
plt.axis("off")


# In[ ]:


plt.imshow(tensor2image(fake_c))
plt.axis("off")

