import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from models import networks
import numpy as np
import os.path as osp
from PIL import Image
import os
import torchvision.transforms as transforms

device = "cuda"
from distributed import (
    get_rank,
    synchronize,
)

parser = argparse.ArgumentParser(description="Training the segment module")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--batchSize", type=int, default=8)
parser.add_argument("--dataroot", type=str, default='viton_hd_dataset')
parser.add_argument("--datapairs", type=str, default='train_pairs.txt')
parser.add_argument("--phase", type=str, default='train')
parser.add_argument("--beta1", type=int, default=0.5)
parser.add_argument("--no_lsgan", type=bool, default=True)

args = parser.parse_args()

print ('Distributed Training Mode.')
torch.distributed.init_process_group(backend="nccl", init_method="env://")
torch.cuda.set_device(args.local_rank)
synchronize()

G1 = networks.PHPM(input_nc=6, output_nc=4).to(device)
D1 = networks.Discriminator(7).to(device)
optimizerG = torch.optim.Adam(G1.parameters(), lr=0.0002, betas=(args.beta1, 0.999))
optimizerD = torch.optim.Adam(D1.parameters(), lr=0.0002, betas=(args.beta1, 0.999))

checkpoint_loc = "checkpoint_segment/"
if get_rank() == 0:
    if not os.path.exists(checkpoint_loc):
        os.makedirs(checkpoint_loc)
    writer = SummaryWriter('runs/segment')

G1 = nn.parallel.DistributedDataParallel(
        G1,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False,
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

        label_path = osp.join(self.opt.dataroot, self.opt.phase, 'candidate_label', h_name+".png")
        label = Image.open(label_path).convert('L')

        dense_path = osp.join(self.opt.dataroot, self.opt.phase, 'candidate_dense', h_name.replace(".jpg", "_iuv.png"))
        dense = np.array(Image.open(dense_path))

        clothes_path = osp.join(self.opt.dataroot, self.opt.phase, 'clothes', c_name)
        clothes = Image.open(clothes_path).convert('RGB')

        clothes_mask_path = osp.join(self.opt.dataroot, self.opt.phase, 'clothes_mask', c_name)
        clothes_mask = Image.open(clothes_mask_path).convert('L')

        transform_A = get_transform(normalize=False)
        transform_B = get_transform()

        label_tensor = transform_A(label) * 255
        label_tensor = torch.nn.functional.pad(input=label_tensor, pad=(82, 82, 0, 0), mode='constant', value=0)
        clothes_tensor = transform_B(clothes)
        clothes_tensor = torch.nn.functional.pad(input=clothes_tensor, pad=(82, 82, 0, 0), mode='constant', value=0)
        clothes_mask_tensor = transform_A(clothes_mask)
        clothes_mask_tensor = torch.nn.functional.pad(input=clothes_mask_tensor, pad=(82, 82, 0, 0), mode='constant', value=0)
        dense_tensor = torch.from_numpy(dense).permute(2, 0, 1)
        dense_tensor = torch.nn.functional.pad(input=dense_tensor, pad=(82, 82, 0, 0), mode='constant', value=0)

        return {'label': label_tensor,'clothes': clothes_tensor,
                'dense': dense_tensor, 'clothes_mask': clothes_mask_tensor}

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

def cross_entropy2d(input, target):
    n, c, h, w = input.size()
    nt, _, ht, wt = target.size()
    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    target = target.type(torch.int64)
    loss = F.cross_entropy(input, target)
    return loss

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

def discriminate(netD ,input_label, real_or_fake):
    input = torch.cat([input_label, real_or_fake], dim=1)
    return netD.forward(input)

criterionGAN = networks.GANLoss(use_lsgan=not args.no_lsgan, tensor=torch.cuda.FloatTensor)

step = 0
sigmoid = nn.Sigmoid()
for epoch in range(50):
    for data in train_dataloader: #training
        clothes = data['clothes'].to(device)
        clothes_mask = data['clothes_mask'].to(device)
        clothes = clothes * clothes_mask
        dense = data['dense'].to(device)
        label = data['label'].float().to(device)

        background_label = (label == 0).float()
        cloth_label = (label == 5).float() + (label == 6).float() + (label == 7).float()
        arm1_label = (label == 14).float()
        arm2_label = (label == 15).float()

        ground_truth_4 = torch.cat([background_label, cloth_label, arm1_label, arm2_label], 1)
        G1_in = torch.cat([clothes, dense], dim=1)
        ground_truth_1 = generate_discrete_label(ground_truth_4.detach(), 4, False)
        requires_grad(G1, False)
        requires_grad(D1, True)

        arm_label = G1(G1_in)
        arm_label = sigmoid(arm_label)
        armlabel_map = generate_discrete_label(arm_label.detach(), 4, False)
        ground_truth_1 = generate_discrete_label(ground_truth_4.detach(), 4, False)

        pred_fake = discriminate(D1, G1_in.detach(), armlabel_map.detach())
        loss_D_fake = criterionGAN(pred_fake, False)
        pred_real = discriminate(D1, G1_in.detach(), ground_truth_1.detach())
        loss_D_real = criterionGAN(pred_real, True)
        loss_D = loss_D_fake + loss_D_real

        optimizerD.zero_grad()
        loss_D.backward()
        optimizerD.step()

        requires_grad(G1, True)
        requires_grad(D1, False)

        arm_label = G1(G1_in)
        arm_label = sigmoid(arm_label)
        armlabel_map = generate_discrete_label(arm_label.detach(), 4, False)
        pred_fake = D1.forward(torch.cat((G1_in.detach(), armlabel_map.detach()), dim=1))
        loss_G_GAN = criterionGAN(pred_fake, True)
        pair_GANloss = loss_G_GAN * 5
        pair_GANloss = pair_GANloss / 2
        CE_loss = cross_entropy2d(arm_label, ground_truth_1)
        loss_G = CE_loss + pair_GANloss

        optimizerG.zero_grad()
        loss_G.backward()
        optimizerG.step()

        step += 1

        if get_rank() == 0:
            if step % 100 == 0:
                writer.add_scalar('loss_G', loss_G, step)
                writer.add_scalar('loss_D', loss_D, step)

            if step % 1000 == 0:
                writer.add_image('generated', torchvision.utils.make_grid(armlabel_map), step)
                writer.add_image('gt', torchvision.utils.make_grid(ground_truth_1), step)
                writer.add_image('clothes', torchvision.utils.make_grid(clothes), step)
                writer.add_image('dense', torchvision.utils.make_grid(dense), step)


    if get_rank() == 0:
        torch.save(G1.module.state_dict(), checkpoint_loc + '/segment_' + str(epoch) + '.pth')
if get_rank() == 0:
    torch.save(G1.module.state_dict(), checkpoint_loc + '/segment_final.pth')
