import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
import tqdm
import cv2
from models import networks, cStyleGAN
import os.path as osp
import torchvision.transforms as transforms
from PIL import Image
from util.coordinate_completion_model import define_G as define_CCM
import os
from util.dp2coor import getSymXYcoordinates

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
    gmm.load_state_dict(torch.load('checkpoint/gmm_mini_loss_49.pth'))
    gmm.eval()

with torch.no_grad():
    G3 = cStyleGAN.GeneratorCustom(size=512, style_dim=2048, n_mlp=8, channel_multiplier=2).cuda()
    G3.load_state_dict(torch.load('checkpoint/pre_9.pt'))
    G3.eval()

coor_completion_generator = define_CCM().cuda()
CCM_checkpoint = torch.load('checkpoint/CCM_epoch50.pt')
coor_completion_generator.load_state_dict(CCM_checkpoint["g"])
coor_completion_generator.eval()
for param in coor_completion_generator.parameters():
    coor_completion_generator.requires_grad = False

transform = get_transform()

def tensor2image(candidate):
    candidate_numpy = (candidate[0].clone() + 1) * 0.5 * 255
    candidate_numpy = candidate_numpy.cpu().clamp(0, 255)
    candidate_numpy = candidate_numpy.detach().numpy().astype('uint8')
    candidate_numpy = candidate_numpy.swapaxes(0, 1).swapaxes(1, 2)
    return candidate_numpy

def correct_colour(garments, correcting_parameters):
    garment = tensor2image(garments)
    img_hsv = cv2.cvtColor(garment, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img_hsv)

    v = cv2.subtract(v, correcting_parameters[0][0].item()) # brightness
    v = cv2.divide(v, correcting_parameters[0][1].item()) # contrast
    h = cv2.subtract(h, correcting_parameters[0][2].item()) # hue
    s = cv2.divide(s, correcting_parameters[0][3].item()) # saturation

    img_hsv = cv2.merge([h, s, v])
    img_result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    img_result = transform(img_result)
    img_result = img_result.unsqueeze(0)

    return img_result.cuda()

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

    in_candidateHD = data['candidateHD'].cuda()
    in_candidateHD = torch.nn.functional.pad(input=in_candidateHD, pad=(82, 82, 0, 0), mode='constant', value=0)
    in_labelHD = data['labelHD'].cuda()
    in_labelHD = torch.nn.functional.pad(input=in_labelHD, pad=(82, 82, 0, 0), mode='constant', value=0)
    in_source_dense = data['source_dense'][0]

    mask_r_arm = (in_labelHD == 14).float()
    mask_l_arm = (in_labelHD == 15).float()
    mask_clothes = (in_labelHD == 5).float()
    mask_dress = (in_labelHD == 6).float()
    mask_jacket = (in_labelHD == 7).float()
    segment_start = mask_clothes + mask_dress + mask_jacket
    skin_color = ger_average_color((mask_r_arm + mask_l_arm - mask_l_arm * mask_r_arm),
                                   (mask_r_arm + mask_l_arm - mask_l_arm * mask_r_arm) * in_candidateHD)
    #img = Image.fromarray(tensor2image(skin_color[:, :, :, int(82):512 - int(82)]))
    #img.save('result/skin_color' + str(step) + '.jpg')
    invisible_torso = skin_color * (mask_r_arm + mask_l_arm + segment_start)
    img_hole_hand = in_candidateHD * (1 - mask_r_arm) * (1 - mask_l_arm) * (1 - segment_start)

    #img = Image.fromarray(tensor2image(img_hole_hand[:, :, :, int(82):512 - int(82)]))
    #img.save('result/img_hole_hand' + str(step) + '.jpg')

    img_hole_hand += invisible_torso

    uv_coor, uv_mask, uv_symm_mask = getSymXYcoordinates(in_source_dense.numpy(), resolution = 512)
    in_denseHD = data['denseHD'].float().cuda()
    in_denseHD = torch.nn.functional.pad(input=in_denseHD, pad=(82, 82, 0, 0), mode='constant', value=0)
    mask_fore = (in_labelHD > 0).float()
    h, w = [512, 348]
    shift = int((h-w)/2) # center shift
    uv_coor[:,:,0] = uv_coor[:,:,0] + shift # put in center
    uv_coor = ((2*uv_coor/(h-1))-1)
    uv_coor = uv_coor*np.expand_dims(uv_mask,2) + (-10*(1-np.expand_dims(uv_mask,2)))
    uv_coor_pytorch = torch.from_numpy(uv_coor).float().permute(2, 0, 1).unsqueeze(0) # from h,w,c to 1,c,h,w
    uv_mask_pytorch = torch.from_numpy(uv_mask).unsqueeze(0).unsqueeze(0).float() #1xchw
    with torch.no_grad():
        coor_completion_generator.eval()
        complete_coor = coor_completion_generator(uv_coor_pytorch.cuda(), uv_mask_pytorch.cuda())

    appearance = torch.cat([img_hole_hand, mask_fore, complete_coor], 1)

    fake_img, _ = G3(appearance=appearance, target_dense=in_denseHD,
                                    segment=target_mask_clothes)
    fake_img *= (1 - target_mask_clothes)
    fake_img += warped_garment
    fake_img = tensor2image(fake_img[:, :, :, int(82):512 - int(82)])
    img = Image.fromarray(fake_img)
    img.save('result_eval/' + name[0])
    #torch.save(pre_clothes_mask, "result_eval/mask_" + name[0])
    #torch.save(in_garmentHD, "result_eval/garment_" + name[0])

    #img.save(str(step) + '.jpg')
    step += 1
    #img = Image.fromarray(tensor2image(target_mask_clothes[:, 0:1, :, int(82):512 - int(82)])[:,:,0], 'L')
    #img.save('result/torso' + str(i) + '.jpg')

    #skin_mask = mask_r_arm + mask_l_arm + segment_start
    #img = Image.fromarray(tensor2image(skin_mask[:, 0:1, :, int(82):512 - int(82)])[:, :, 0], 'L')
    #img.save('result/skin_mask' + str(i) + '.jpg')
