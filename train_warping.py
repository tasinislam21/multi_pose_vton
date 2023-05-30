import os
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from models import networks
import os.path as osp
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from models.dual_disc import MultiscaleDiscriminator, GANLoss

device = "cuda"
from distributed import (
    get_rank,
    synchronize
)

parser = argparse.ArgumentParser(description="training the warping module")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--batchSize", type=int, default=8)
parser.add_argument("--dataroot", type=str, default='viton_hd_dataset')
parser.add_argument("--datapairs", type=str, default='train_pairs.txt')
parser.add_argument("--phase", type=str, default='train')

args = parser.parse_args()

print ('Distributed Training Mode.')
torch.distributed.init_process_group(backend="nccl", init_method="env://")
torch.cuda.set_device(args.local_rank)
synchronize()

G1 = networks.GMM(input_nc=7).to(device)
discriminator = MultiscaleDiscriminator().to(device)

optimizerG = torch.optim.Adam(G1.parameters(), lr=0.0001, betas=(0, 0.9))
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0, 0.9))

checkpoint_loc = "checkpoint_warping/"
if get_rank() == 0:
    if not os.path.exists('checkpoint_loc'):
        os.makedirs('checkpoint_loc')
    writer = SummaryWriter('runs/warping')

G1 = nn.parallel.DistributedDataParallel(
        G1,
        find_unused_parameters=True,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False
    )

D1 = nn.parallel.DistributedDataParallel(
        discriminator,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False,
    )

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

inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean_clothing, std_clothing)],
    std=[1/s for s in std_clothing]
)

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
        with open(os.path.join(opt.dataroot, opt.datapairs), 'r') as f:
            for line in f.readlines():
                h_name, _ = line.strip().split()
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

train_dataset = BaseDataset(args)
sampler = data_sampler(train_dataset, shuffle=True, distributed=True)

train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batchSize,
            sampler=sampler,
            drop_last=True,
            pin_memory=True,
            num_workers=2)

criterionGAN = GANLoss('hinge', tensor=torch.cuda.FloatTensor)
criterionVGG = networks.VGGLoss()
criterionFeat = nn.L1Loss()

tanh = nn.Tanh()

step = 0

for epoch in range(100):
    for data in train_dataloader: #training
        candidate = data['candidate'].to(device)
        clothes = data['clothes'].to(device)
        clothes_mask = data['clothes_mask'].to(device)
        clothes = clothes * clothes_mask
        skeleton = data['skeleton'].to(device)
        label = data['label'].float().to(device)
        cloth_label = (label == 5).float() + (label == 6).float() + (label == 7).float()
        ground_truth = candidate * cloth_label

        with torch.no_grad():
            fake_c, _ = G1.forward(clothes, cloth_label, skeleton)
            fake_c = tanh(fake_c)
            fake_c *= cloth_label
            fake_c = fake_c.detach()
            fake_c.requires_grad_()

        fake_concat = torch.cat((skeleton, cloth_label, fake_c), dim=1)
        real_concat = torch.cat((skeleton, cloth_label, ground_truth), dim=1)
        pred = discriminator(torch.cat((fake_concat, real_concat), dim=0))
        D_losses = {}
        if type(pred) == list:
            pred_fake = []
            pred_real = []
            for p in pred:
                pred_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                pred_real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            pred_fake = pred[:pred.size(0) // 2]
            pred_real = pred[pred.size(0) // 2:]

        D_losses['D_Fake'] = criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_Real'] = criterionGAN(pred_real, True, for_discriminator=True)
        loss_dis = sum(D_losses.values()).mean()

        optimizerD.zero_grad()
        loss_dis.backward()
        optimizerD.step()

        fake_c, affine = G1.forward(clothes, cloth_label, skeleton)
        fake_c = tanh(fake_c)
        fake_c *= cloth_label

        fake_concat = torch.cat((skeleton, cloth_label, fake_c), dim=1)
        real_concat = torch.cat((skeleton, cloth_label, ground_truth), dim=1)
        pred = discriminator(torch.cat((fake_concat, real_concat), dim=0))
        G_losses = {}
        if type(pred) == list:
            pred_fake = []
            pred_real = []
            for p in pred:
                pred_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                pred_real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            pred_fake = pred[:pred.size(0) // 2]
            pred_real = pred[pred.size(0) // 2:]

        G_losses['GAN'] = criterionGAN(pred_fake, True, for_discriminator=False)
        num_D = len(pred_fake)
        GAN_Feat_loss = torch.cuda.FloatTensor(1).zero_()
        for i in range(num_D):  # for each discriminator
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                GAN_Feat_loss += unweighted_loss * 10.0 / num_D
        G_losses['GAN_Feat'] = GAN_Feat_loss
        G_losses['VGG'] = criterionVGG(fake_c, ground_truth) * 10.0
        G_losses['affine_loss'] = criterionFeat(affine, ground_truth)
        loss_gen = sum(G_losses.values()).mean()

        optimizerG.zero_grad()
        loss_gen.backward()
        optimizerG.step()

        step += 1

        if get_rank() == 0:
            if step % 100 == 0:
                writer.add_scalar('GAN_loss', G_losses['GAN'], step)
                writer.add_scalar('GAN_Feat_loss', G_losses['GAN_Feat'], step)
                writer.add_scalar('VGG_loss', G_losses['VGG'], step)
                writer.add_scalar('overall_g', loss_gen, step)

                writer.add_scalar('GAN_loss', D_losses['D_Fake'], step)
                writer.add_scalar('GAN_Feat_loss', D_losses['D_Real'], step)
                writer.add_scalar('overall_d', loss_dis, step)

            if step % 200 == 0:
                writer.add_image('gt', torchvision.utils.make_grid(gt_normalize(ground_truth)), step)
                writer.add_image('affine_garment', torchvision.utils.make_grid(inv_normalize(affine)), step)
                writer.add_image('generated', torchvision.utils.make_grid(gt_normalize(fake_c)), step)

    if get_rank() == 0:
        torch.save(G1.module.state_dict(), checkpoint_loc + '/gmm_' + str(epoch) + '.pth')
if get_rank() == 0:
    torch.save(G1.module.state_dict(), checkpoint_loc + '/gmm_final.pth')