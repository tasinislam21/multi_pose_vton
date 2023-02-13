import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from models import networks, xing
import os.path as osp
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,5,6,7"

device = "cuda"
from distributed import (
    get_rank,
    synchronize
)

class Args:
    batchSize = 2
    dataroot = 'data'
    datapairs = 'train_pairs.txt'
    phase = 'train'
    beta1 = 0.5
    no_lsgan = True
    pool_size = 50

opt = Args

parser = argparse.ArgumentParser(description="Pose with Style trainer")
parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
args = parser.parse_args()

print ('Distributed Training Mode.')
torch.distributed.init_process_group(backend="nccl", init_method="env://")
torch.cuda.set_device(args.local_rank)
synchronize()

#G1 = networks.GMM(input_nc=7).to(device)
G1 = xing.XingNetwork(input_nc=[3,3], output_nc=3).to(device)
#D1 = networks.Discriminator(7).to(device)
D1 = xing.ResnetDiscriminator(input_nc=6, use_dropout=True, n_blocks=3, use_sigmoid=True).to(device)

optimizerG = torch.optim.Adam(G1.parameters(), lr=0.0002, betas=(opt.beta1, 0.999))
optimizerD = torch.optim.Adam(D1.parameters(), lr=0.0002, betas=(opt.beta1, 0.999))

if get_rank() == 0:
    if not os.path.exists('gmm'):
        os.makedirs('gmm')
    writer = SummaryWriter('runs/xing')

G1 = nn.parallel.DistributedDataParallel(
        G1,
        find_unused_parameters=True,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False
    )

D1 = nn.parallel.DistributedDataParallel(
        D1,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False,
    )

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
        human_names = []
        cloth_names = []
        with open(os.path.join(opt.dataroot, opt.datapairs), 'r') as f:
            for line in f.readlines():
                h_name, _ = line.strip().split()
                human_names.append(h_name)
                cloth_names.append(h_name)
        self.human_names = human_names
        self.cloth_names = cloth_names

    def __getitem__(self, index):
        c_name = self.cloth_names[index]
        h_name = self.human_names[index]

        candidate_path = osp.join(self.opt.dataroot, self.opt.phase, 'candidate', h_name)
        candidate = Image.open(candidate_path).convert('RGB')

        label_path = osp.join(self.opt.dataroot, self.opt.phase, 'candidate_label', h_name+".png")
        label = Image.open(label_path).convert('L')

        skeleton_path = osp.join(self.opt.dataroot, self.opt.phase, 'candidate_skeleton', h_name.replace(".jpg", "_rendered.png"))
        skeleton = Image.open(skeleton_path).convert('RGB')

        dense_path = osp.join(self.opt.dataroot, self.opt.phase, 'candidate_dense', h_name.replace(".jpg", "_iuv.png"))
        dense = np.array(Image.open(dense_path))

        clothes_path = osp.join(self.opt.dataroot, self.opt.phase, 'clothes', c_name)
        clothes = Image.open(clothes_path).convert('RGB')

        clothes_mask_path = osp.join(self.opt.dataroot, self.opt.phase, 'clothes_mask', c_name)
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
                'skeleton': skeleton_tensor, 'clothes_mask': clothes_mask_tensor, 'dense': dense_tensor}

    def __len__(self):
        return len(self.human_names)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

train_dataset = BaseDataset(opt)
sampler = data_sampler(train_dataset, shuffle=True, distributed=True)

train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batchSize,
            sampler=sampler,
            drop_last=True,
            pin_memory=True,
            num_workers=2)

def discriminate(netD ,input_label, real_or_fake):
    input = torch.cat([input_label, real_or_fake], dim=1)
    return netD.forward(input)

def backward_D_basic(netD, real, fake):
    # Real
    pred_real = netD(real)
    loss_D_real = criterionGAN(pred_real, True) * 5.0
    # Fake
    pred_fake = netD(fake.detach())
    loss_D_fake = criterionGAN(pred_fake, False) * 5.0
    # Combined loss
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    # backward
    loss_D.backward()
    return loss_D

criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=torch.cuda.FloatTensor)
criterionVGG = networks.VGGLoss()
criterionFeat = nn.L1Loss()

tanh = nn.Tanh()
checkpoint_loc = "gmm/"

step = 0

g_module = G1.module

for epoch in range(50):
    for data in train_dataloader: #training
        candidate = data['candidate'].to(device)
        clothes = data['clothes'].to(device)
        clothes_mask = data['clothes_mask'].to(device)
        clothes = clothes * clothes_mask
        skeleton = data['skeleton'].to(device)
        label = data['label'].float().to(device)
        cloth_label = (label == 5).float() + (label == 6).float() + (label == 7).float()

        requires_grad(D1, True)
        requires_grad(G1, False)

        #fake_c, affine = G1.forward(clothes, cloth_label, skeleton)
        fake_c, affine = G1(input=[clothes, cloth_label, skeleton])
        fake_c *= cloth_label
        fake_c = tanh(fake_c)

        real_pool = (candidate * cloth_label)
        fake_pool = fake_c
        input_pool = torch.cat([cloth_label, clothes], 1)
        D_pool = D1

        #pred_fake = discriminate(D_pool, input_pool.detach(), fake_pool.detach())
        #loss_D_fake = criterionGAN(pred_fake, False)
        #pred_real = discriminate(D_pool, input_pool.detach(), real_pool.detach())
        #loss_D_real = criterionGAN(pred_real, True)
        #loss_D = (loss_D_fake + loss_D_real)

        loss_D_PP = backward_D_basic(D1, torch.cat([real_pool, skeleton], 1), torch.cat([fake_pool, skeleton], 1))

        optimizerD.zero_grad()
        #loss_D.backward()
        loss_D = loss_D_PP.item()
        optimizerD.step()

        requires_grad(D1, False)
        requires_grad(G1, True)

        #fake_c, affine = G1.forward(clothes, cloth_label, skeleton)
        fake_c, affine = G1(input=[clothes, cloth_label, skeleton])
        fake_c *= cloth_label
        fake_c = tanh(fake_c)

        real_pool = (candidate * cloth_label)
        fake_pool = fake_c
        #input_pool = torch.cat([cloth_label, clothes], 1)
        D_pool = D1

        #pred_fake = D_pool.forward(torch.cat((input_pool.detach(), fake_pool.detach()), dim=1))
        pred_fake = D1(torch.cat([fake_pool, skeleton], 1).detach())
        loss_G_GAN = criterionGAN(pred_fake, True)
        loss_G_VGG = criterionVGG(fake_pool, real_pool)
        L1_loss = criterionFeat(affine, real_pool)
        L1_loss = criterionFeat(fake_pool, real_pool)
        loss_G = loss_G_GAN + L1_loss + loss_G_VGG

        optimizerG.zero_grad()
        loss_G.backward()
        optimizerG.step()

        step += 1

        if get_rank() == 0:
            if step % 100 == 0:
                writer.add_scalar('loss_G', loss_G, step)
                writer.add_scalar('loss_D', loss_D, step)

            if step % 200 == 0:
                writer.add_image('clothes', torchvision.utils.make_grid(clothes), step)
                writer.add_image('gt', torchvision.utils.make_grid(candidate*cloth_label), step)
                writer.add_image('affine_garment', torchvision.utils.make_grid(affine), step)
                writer.add_image('generated', torchvision.utils.make_grid(fake_c), step)

    #for param_group in optimizerD.param_groups:
    #    param_group['lr'] *= 0.75
    if get_rank() == 0:
        torch.save(g_module.state_dict(), checkpoint_loc + '/gmm_xing_' + str(epoch) + '.pth')
if get_rank() == 0:
    torch.save(g_module.state_dict(), checkpoint_loc + '/gmm_xing_final.pth')