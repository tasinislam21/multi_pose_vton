import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from models import networks
import os.path as osp
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np

class Args:
    batchSize = 1
    dataroot = 'data'
    datapairs = 'test_shuffle.txt'
    phase = 'test'

opt = Args

with torch.no_grad():
    G1 = networks.GMM(input_nc=7, output_nc=4).cuda()
    G1.load_state_dict(torch.load('checkpoint/ghost_gmm_49.pth'))
    G1.eval()


def get_transform(normalize=True):
    transform_list = []
    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class BaseDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        super(BaseDataset, self).__init__()
        self.human_names = []
        self.pose_names = []
        self.cloth_names = []
        with open(os.path.join(opt.dataroot, opt.datapairs), 'r') as f:
            for line in f.readlines():
                h_name, p_name, c_name = line.strip().split()
                self.human_names.append(h_name)
                self.pose_names.append(p_name)
                #self.cloth_names.append(c_name)
                self.cloth_names.append(h_name)

    def __getitem__(self, index):
        c_name = self.cloth_names[index]
        h_name = self.human_names[index]
        p_name = self.pose_names[index]

        candidate_path = osp.join(self.opt.dataroot, self.opt.phase, 'candidateHD', h_name)
        candidate = Image.open(candidate_path).convert('RGB')

        label_path = osp.join(self.opt.dataroot, self.opt.phase, 'candidateHD_label', p_name.replace("_iuv.png", ".jpg.png"))
        label = Image.open(label_path).convert('L')

        skeleton_path = osp.join(self.opt.dataroot, self.opt.phase, 'candidateHD_pose', p_name.replace("_iuv.png", "_rendered.png"))
        skeleton = Image.open(skeleton_path).convert('RGB')

        dense_path = osp.join(self.opt.dataroot, self.opt.phase, 'candidateHD_dense', p_name)
        dense = np.array(Image.open(dense_path))

        clothes_path = osp.join(self.opt.dataroot, self.opt.phase, 'clothesHD', c_name)
        clothes = Image.open(clothes_path).convert('RGB')

        clothes_mask_path = osp.join(self.opt.dataroot, self.opt.phase, 'clothesHD_mask', c_name)
        clothes_mask = Image.open(clothes_mask_path).convert('L')

        transform_A = get_transform(normalize=False)
        transform_B = get_transform()

        candidate_tensor = transform_B(candidate)
        candidate_tensor = torch.nn.functional.pad(input=candidate_tensor, pad=(82, 82, 0, 0), mode='constant', value=0)
        label_tensor = transform_A(label) * 255
        label_tensor = torch.nn.functional.pad(input=label_tensor, pad=(82, 82, 0, 0), mode='constant', value=0)
        clothes_tensor = transform_B(clothes)
        clothes_tensor = torch.nn.functional.pad(input=clothes_tensor, pad=(82, 82, 0, 0), mode='constant', value=0)
        clothes_mask_tensor = transform_A(clothes_mask)
        clothes_mask_tensor = torch.nn.functional.pad(input=clothes_mask_tensor, pad=(82, 82, 0, 0), mode='constant', value=0)
        skeleton_tensor = transform_B(skeleton)
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
            batch_size=opt.batchSize)


device = 'cuda'
tanh = nn.Tanh()
sigmoid = nn.Sigmoid()

def tensor2image(tensor):
    tensor = (tensor[0].clone() + 1) * 0.5 * 255
    tensor = tensor.cpu().clamp(0, 255)
    array = tensor.detach().numpy().astype('uint8')
    array = array.swapaxes(0, 1).swapaxes(1, 2)
    return array

for data in train_dataloader: #training
    candidate = data['candidate'].to(device)
    clothes = data['clothes'].to(device)
    clothes_mask = data['clothes_mask'].to(device)
    clothes = clothes * clothes_mask
    skeleton = data['skeleton'].to(device)
    label = data['label'].float().to(device)
    cloth_label = (label == 5).float() + (label == 6).float() + (label == 7).float()

    fake_c, affine = G1.forward(clothes, cloth_label, skeleton)
    composition_mask = fake_c[:, 3:4, :, :]
    composition_mask = sigmoid(composition_mask)
    fake_c = fake_c[:, 0:3, :, :]
    fake_c *= cloth_label
    fake_c = tanh(fake_c)

    comp_fake_c = fake_c.detach() * (1 - composition_mask) + (
        composition_mask) * affine.detach()

    img = Image.fromarray(tensor2image(comp_fake_c[:, :, :, int(82):512 - int(82)]))
    name = data['name']
    img.save('ghost/' + name[0])