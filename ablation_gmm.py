import os
from os import listdir
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from models import networks
import os.path as osp
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

device = "cuda"

class Args:
    batchSize = 1
    dataroot = 'data'
    phase = 'test'

opt = Args

with torch.no_grad():
    G1 = networks.GMM(input_nc=7, output_nc=3)
    G1.cuda()
    G1.load_state_dict(torch.load('checkpoint/gmm_final.pth'))
    G1.eval()

mean_clothing = [0.5149, 0.5003, 0.4985]
std_clothing = [0.4498, 0.4467, 0.4442]

mean_candidate = [0.4998, 0.4790, 0.4719]
std_candidate = [0.4147, 0.4081, 0.4063]

mean_skeleton = [0.0101, 0.0082, 0.0040]
std_skeleton = [0.0716, 0.0630, 0.0426]


def get_transform(normalize=True, mean=None, std=None):
    transform_list = []
    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(transform_list)

gt_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean_candidate, std_candidate)],
    std=[1/s for s in std_candidate]
)

class BaseDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        super(BaseDataset, self).__init__()
        human_names = []
        cloth_names = []
        files = [f for f in listdir(os.path.join(opt.dataroot, opt.phase, "candidateHD"))]
        for h_name in files:
            human_names.append(h_name)
            cloth_names.append(h_name)
        self.human_names = human_names
        self.cloth_names = cloth_names
        self.transform_Mask = get_transform(normalize=False)
        self.transform_Clothes = get_transform(mean=mean_clothing, std=std_clothing)
        self.transform_Candidate = get_transform(mean=mean_candidate, std=std_candidate)
        self.transform_Skeleton = get_transform(mean=mean_skeleton, std=std_skeleton)

    def __getitem__(self, index):
        c_name = self.cloth_names[index]
        h_name = self.human_names[index]

        candidate_path = osp.join(self.opt.dataroot, self.opt.phase, 'candidateHD', h_name)
        candidate = Image.open(candidate_path).convert('RGB')

        label_path = osp.join(self.opt.dataroot, self.opt.phase, 'candidateHD_label', h_name+".png")
        label = Image.open(label_path).convert('L')

        skeleton_path = osp.join(self.opt.dataroot, self.opt.phase, 'candidateHD_pose', h_name.replace(".jpg", "_rendered.png"))
        skeleton = Image.open(skeleton_path).convert('RGB')

        dense_path = osp.join(self.opt.dataroot, self.opt.phase, 'candidateHD_dense', h_name.replace(".jpg", "_iuv.png"))
        dense = np.array(Image.open(dense_path))

        clothes_path = osp.join(self.opt.dataroot, self.opt.phase, 'clothesHD', c_name)
        clothes = Image.open(clothes_path).convert('RGB')

        clothes_mask_path = osp.join(self.opt.dataroot, self.opt.phase, 'clothesHD_mask', c_name)
        clothes_mask = Image.open(clothes_mask_path).convert('L')

        candidate_tensor = self.transform_Candidate(candidate)
        candidate_tensor = torch.nn.functional.pad(input=candidate_tensor, pad=(82, 82, 0, 0), mode='constant', value=0)
        label_tensor = self.transform_Mask(label) * 255
        label_tensor = torch.nn.functional.pad(input=label_tensor, pad=(82, 82, 0, 0), mode='constant', value=0)
        clothes_tensor = self.transform_Clothes(clothes)
        clothes_tensor = torch.nn.functional.pad(input=clothes_tensor, pad=(82, 82, 0, 0), mode='constant', value=0)
        clothes_mask_tensor = self.transform_Mask(clothes_mask)
        clothes_mask_tensor = torch.nn.functional.pad(input=clothes_mask_tensor, pad=(82, 82, 0, 0), mode='constant', value=0)
        skeleton_tensor = self.transform_Skeleton(skeleton)
        skeleton_tensor = torch.nn.functional.pad(input=skeleton_tensor, pad=(82, 82, 0, 0), mode='constant', value=0)

        dense_tensor = torch.from_numpy(dense).permute(2, 0, 1)
        dense_tensor = torch.nn.functional.pad(input=dense_tensor, pad=(82, 82, 0, 0), mode='constant', value=0)

        return {'label': label_tensor,'clothes': clothes_tensor, 'candidate': candidate_tensor,
                'skeleton': skeleton_tensor, 'clothes_mask': clothes_mask_tensor, 'dense': dense_tensor,
                'name': h_name}

    def __len__(self):
        return len(self.human_names)


train_dataset = BaseDataset(opt)
train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batchSize,
            num_workers=0)

tanh = nn.Tanh()
step = 0

def tensor2image(tensor_clothing):
    numpy_clothing = tensor_clothing[0].cpu().detach().numpy()
    numpy_clothing = (numpy_clothing * 255).astype(np.uint8)
    numpy_clothing = numpy_clothing.transpose(1, 2, 0)
    image_pil = Image.fromarray(numpy_clothing)
    return image_pil

for data in train_dataloader:
    candidate = data['candidate'].to(device)
    clothes = data['clothes'].to(device)
    clothes_mask = data['clothes_mask'].to(device)
    clothes = clothes * clothes_mask
    skeleton = data['skeleton'].to(device)
    label = data['label'].float().to(device)
    cloth_label = (label == 5).float() + (label == 6).float() + (label == 7).float()
    ground_truth = candidate * cloth_label
    name = data['name'][0]

    fake_c, _ = G1.forward(clothes, cloth_label, skeleton)
    fake_c = tanh(fake_c)
    fake_c *= cloth_label

    fake_img = tensor2image(gt_normalize(fake_c[:, :, :, int(82):512 - int(82)]))
    #real_img = tensor2image(gt_normalize(ground_truth))

    fake_img.save('ablation/fake_disc1/' + str(name))
    #real_img.save('ablation/real/' + str(name))