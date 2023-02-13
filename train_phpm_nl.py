import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from models import networks, dataset
import numpy as np
import tqdm
from util.image_pool import ImagePool
import os

writer = SummaryWriter('runs/phpm_nl')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Args:
    batchSize = 32
    dataroot = 'data'
    datapairs = 'train_pairs.txt'
    phase = 'train'
    beta1 = 0.5
    gpu_ids = [i for i in range(torch.cuda.device_count())]
    no_lsgan = True
    pool_size = 50


opt_train = Args
G1 = networks.PHPM_NL(6, 4).cuda()
D1 = networks.ResnetDiscriminator(input_nc=4, use_dropout=True, n_blocks=3,
            gpu_ids=opt_train.gpu_ids, use_sigmoid=True).cuda()

train_dataset = dataset.BaseDataset(opt_train)
train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt_train.batchSize,
            num_workers=4)

def cross_entropy2d(input, target, weight=None, size_average=True):
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

def backward_D_basic(netD, real, fake):
    # Real
    pred_real = nn.parallel.data_parallel(netD, real, opt_train.gpu_ids)
    loss_D_real = criterionGAN(pred_real, True) * 5.0
    # Fake
    pred_fake = nn.parallel.data_parallel(netD, fake.detach(), opt_train.gpu_ids)
    loss_D_fake = criterionGAN(pred_fake, False) * 5.0
    # Combined loss
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    # backward
    loss_D.backward()
    return loss_D

criterionGAN = networks.GANLoss(use_lsgan=not opt_train.no_lsgan, tensor=torch.cuda.FloatTensor)
fake_PP_pool = ImagePool(opt_train.pool_size)

sigmoid = nn.Sigmoid()
optimizerG = torch.optim.AdamW(G1.parameters(), lr=0.0002, betas=(opt_train.beta1, 0.999))
optimizerD = torch.optim.AdamW(D1.parameters(), lr=0.00015, betas=(opt_train.beta1, 0.999))
checkpoint_loc = "checkpoints/"

step = 0
G1.train()
D1.train()

for epoch in range(50):
    for data in tqdm.tqdm(train_dataloader): #training
        in_label = data['Label'].cuda()
        in_gt1 = data['GT'].cuda()
        in_gt4 = data['GT4'].cuda()
        input_P1 = data['Image'].cuda()

        mask_fore = (in_label > 0).long()
        img_fore = (input_P1 * mask_fore).cuda()
        size = in_label.size()

        clothes = data['Cloth'].cuda()
        skeleton = data['Skeleton'].cuda()
        G1_input = torch.cat([clothes, skeleton], 1)
        arm_label = nn.parallel.data_parallel(G1, G1_input, opt_train.gpu_ids)
        arm_label = sigmoid(arm_label)

        optimizerG.zero_grad()
        pred_fake_PP = nn.parallel.data_parallel(D1, (arm_label), opt_train.gpu_ids)
        loss_G_GAN_PP = criterionGAN(pred_fake_PP, True)
        pair_GANloss = loss_G_GAN_PP * 5
        pair_GANloss = pair_GANloss / 2
        CE_loss = cross_entropy2d(arm_label, in_gt1) * 10
        loss_G = CE_loss + pair_GANloss
        loss_G.backward()
        optimizerG.step()

        optimizerD.zero_grad()
        real_PP = in_gt4
        fake_PP = fake_PP_pool.query(arm_label.data)
        loss_D_PP = backward_D_basic(D1, real_PP, fake_PP) * 10
        loss_D_PP = loss_D_PP.item()
        optimizerD.step()

        if step % 300 == 0:
            writer.add_scalar('loss_G', loss_G, step)
            writer.add_scalar('loss_D', loss_D_PP, step)

        if step % 1000 == 0:
            armlabel_map = generate_discrete_label(arm_label.detach(), 4, False)
            writer.add_image('generated', torchvision.utils.make_grid(armlabel_map), step)
            writer.add_image('gt', torchvision.utils.make_grid(in_gt1), step)
        step += 1

    torch.save(G1.state_dict(), checkpoint_loc + '/phpm_nl_' + str(epoch) + '.pth')
torch.save(G1.state_dict(), checkpoint_loc + '/phpm_nl_final.pth')