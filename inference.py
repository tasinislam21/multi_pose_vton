import os
#os.environ["CUDA_VISIBLE_DEVICES"]="5"
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
import tqdm
from models import networks, cStyleGAN
import os.path as osp
import torchvision.transforms as transforms
from PIL import Image
from util.coordinate_completion_model import define_G as define_CCM
from util.dp2coor import getSymXYcoordinates
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int, default=1)
parser.add_argument("--datapairs", type=str, default='test_shuffle.txt')
parser.add_argument("--dataroot", type=str, default='viton_hd_dataset')
parser.add_argument("--phase", type=str, default='test')
opt = parser.parse_args()

mean_clothing = [0.5149, 0.5003, 0.4985]
std_clothing = [0.4498, 0.4467, 0.4442]

mean_candidate = [0.5, 0.5, 0.5]
std_candidate = [0.5, 0.5, 0.5]

mean_skeleton = [0.0101, 0.0082, 0.0040]
std_skeleton = [0.0716, 0.0630, 0.0426]

inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean_clothing, std_clothing)],
    std=[1/s for s in std_clothing]
)

candidate_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip([0.4998, 0.4790, 0.4719], [0.4147, 0.4081, 0.4063])],
    std=[1/s for s in [0.4147, 0.4081, 0.4063]]
)

def get_transform(normalize=True, mean=None, std=None):
    transform_list = []
    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(transform_list)

class KeyDataset(data.Dataset):
    def __init__(self):
        super(KeyDataset, self).__init__()
        self.transform_Mask = get_transform(normalize=False)
        self.transform_Clothes = get_transform(mean=mean_clothing, std=std_clothing)
        self.transform_Candidate = get_transform(mean=mean_candidate, std=std_candidate)
        self.transform_Skeleton = get_transform(mean=mean_skeleton, std=std_skeleton)

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_candidateHD = os.path.join(opt.dataroot, opt.phase, 'candidateHD')
        self.dir_LabelHD = os.path.join(opt.dataroot, opt.phase, 'candidateHD_label')
        self.dir_denseHD = os.path.join(opt.dataroot, opt.phase, 'candidateHD_dense')
        self.dir_poseHD = osp.join(opt.dataroot, opt.phase, 'candidateHD_pose')
        self.dir_clothesHD = osp.join(opt.dataroot, opt.phase, 'clothesHD')
        self.dir_clothesMaskHD = osp.join(opt.dataroot, opt.phase, 'clothesHD_mask')
        self.dir_warped_person = osp.join(opt.dataroot, opt.phase, 'warped_person')

        self.init_categories(opt.dataroot ,opt.datapairs)


    def init_categories(self, dataroot, pairLst):
        self.human_names = []
        self.pose_names = []
        self.cloth_names = []
        with open(os.path.join(dataroot, pairLst), 'r') as f:
            for line in f.readlines():
                h_name, p_name, c_name = line.strip().split()
                self.human_names.append(h_name)
                self.pose_names.append(p_name)
                self.cloth_names.append(c_name)
                #self.cloth_names.append(h_name)

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
        warped_person_path = osp.join(self.dir_warped_person, candidate_name[:-3]+'pt')
        candidateHD_img = Image.open(candidateHD_path).convert('RGB')
        labelHD_img = Image.open(labelHD_path).convert('L')
        poseHD_img = Image.open(poseHD_path).convert('RGB')
        clothesHD_img = Image.open(clothesHD_path).convert('RGB')
        clothesHD_mask_img = Image.open(clothesHD_mask_path).convert('L')

        denseHD = np.array(Image.open(denseHD_path))
        source_denseHD = np.array(Image.open(source_dense_path))

        candidateHD = self.transform_Candidate(candidateHD_img)
        labelHD = self.transform_Mask(labelHD_img) * 255
        poseHD = self.transform_Skeleton(poseHD_img)
        clothesHD = self.transform_Clothes(clothesHD_img)
        denseHD = torch.from_numpy(denseHD).permute(2, 0, 1)
        clothesHD_mask = self.transform_Mask(clothesHD_mask_img)
        warped_person = torch.load(warped_person_path)
        return {'candidateHD': candidateHD, 'labelHD': labelHD, 'denseHD': denseHD, 'source_dense': source_denseHD,
                'clothesHD': clothesHD, 'clothesHD_mask': clothesHD_mask, 'poseHD': poseHD,
                'warped_person': warped_person, 'name':candidate_name}

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
    G1n.load_state_dict(torch.load('checkpoint/segment.pth'))
    G1n.eval()

with torch.no_grad():
    gmm = networks.GMM(input_nc=7, output_nc=3)
    gmm.cuda()
    gmm.load_state_dict(torch.load('checkpoint/warping.pth'))
    gmm.eval()

with torch.no_grad():
    G3 = cStyleGAN.GeneratorCustom(size=512, style_dim=2048, n_mlp=8, channel_multiplier=2).cuda()
    G3.load_state_dict(torch.load('checkpoint/pose_transfer.pt'))
    G3.eval()

coor_completion_generator = define_CCM().cuda()
CCM_checkpoint = torch.load('checkpoint/CCM_epoch50.pt')
coor_completion_generator.load_state_dict(CCM_checkpoint["g"])
coor_completion_generator.eval()
for param in coor_completion_generator.parameters():
    coor_completion_generator.requires_grad = False

transform = get_transform()

def tensor2image(tensor_candidate, tensor_clothing, mask):
    tensor_candidate = (tensor_candidate[0].clone() + 1) * 0.5 * 255
    tensor_candidate = tensor_candidate.cpu().clamp(0, 255)
    tensor_candidate *= 1-mask[0].cpu().detach()
    numpy_candidate = tensor_candidate.detach().numpy().astype('uint8')
    numpy_candidate = numpy_candidate.swapaxes(0, 1).swapaxes(1, 2)

    tensor_clothing *= mask
    numpy_clothing = tensor_clothing[0].cpu().detach().numpy()
    numpy_clothing = (numpy_clothing * 255).astype(np.uint8)
    numpy_clothing = numpy_clothing.transpose(1, 2, 0)

    numpy_final = numpy_clothing + numpy_candidate
    image_pil = Image.fromarray(numpy_final)
    return image_pil

checkpoint_loc = "result/"
if not os.path.exists(checkpoint_loc):
    os.makedirs(checkpoint_loc)

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
    warped_garment = tanh(warped_garment)
    warped_garment = candidate_normalize(warped_garment)

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
    invisible_torso = skin_color * (mask_r_arm + mask_l_arm + segment_start)
    img_hole_hand = in_candidateHD * (1 - mask_r_arm) * (1 - mask_l_arm) * (1 - segment_start)
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

    img = tensor2image(tensor_candidate=fake_img[:, :, :, int(82):512 - int(82)],
                       tensor_clothing=warped_garment[:, :, :, int(82):512 - int(82)],
                       mask=target_mask_clothes[:, :, :, int(82):512 - int(82)])
    img.save(checkpoint_loc + str(name[0]))