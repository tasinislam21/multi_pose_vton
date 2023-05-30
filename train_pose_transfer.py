import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4"
import sphereface
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms
from tqdm import tqdm
from dataset import CustomDeepFashionDatasetHD
from models.cStyleGAN import GeneratorCustom, VGGLoss, Discriminator, GANLoss
from tensorboardX import SummaryWriter
import torchvision

from distributed import (
    get_rank,
    synchronize,
)
from op import conv2d_gradfix

criterionGAN = GANLoss(use_lsgan=False, tensor=torch.cuda.FloatTensor)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def tensor2image(tensor):
    tensor = (tensor.clone() + 1) * 0.5 * 255
    tensor = tensor.cpu().clamp(0, 255)
    array = tensor.detach().numpy().astype('uint8')
    array = array.swapaxes(0, 1).swapaxes(1, 2)
    return array

def get_transform():
    transform_list = [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
transform = get_transform()

def getFace(images, FT, LP, RP):
    faces = []
    b, h, w, c = images.shape
    for b in range(images.shape[0]):
        if not (abs(FT[b]).sum() == 0):
            current_im = images[b][:, :, int(RP[b].item()):w-int(LP[b].item())].unsqueeze(0)
            theta = FT[b].unsqueeze(0)[:, :2] #bx2x3
            grid = torch.nn.functional.affine_grid(theta, (1, 3, 112, 96))
            current_face = torch.nn.functional.grid_sample(current_im, grid)
            faces.append(current_face)
    if len(faces) == 0:
        return None
    return torch.cat(faces, 0)

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

def discriminate(netD ,input_label, real_or_fake):
    input = torch.cat([input_label, real_or_fake], dim=1)
    return netD.forward(input)

def train(args, loader, sampler, generator, discriminator, g_optim, d_optim, g_ema, device):
    pbar = range(args.epoch)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_epoch, dynamic_ncols=True, smoothing=0.01)
        pbar.set_description('Epoch Counter')

    criterionL1 = torch.nn.L1Loss()
    criterionVGG = VGGLoss(device).to(device)
    if args.faceloss:
        criterionCOS = nn.CosineSimilarity()

    g_module = generator.module
    d_module = discriminator.module

    accum = 0.5 ** (32 / (10 * 1000))

    step = -1
    rank_number = get_rank()
    for idx in pbar:
        epoch = idx + args.start_epoch

        if epoch > args.epoch:
            print("Done!")
            break

        if args.distributed:
            sampler.set_epoch(epoch)

        #batch_time = AverageMeter()
        #####################################
        ############ START EPOCH ############
        #####################################
        for i, data in enumerate(loader):
            step += 1
            input_P1 = data['P1'].cuda()
            input_P2 = data['P2'].cuda()
            in_label = data['Label'].cuda()
            in_label2 = data['Label2'].cuda()
            target_dense = data['target_dense'].float().cuda()

            mask_r_arm = (in_label == 14).float()
            mask_l_arm = (in_label == 15).float()
            mask_clothes = (in_label == 5).float()
            mask_dress = (in_label == 6).float()
            mask_jacket = (in_label == 7).float()
            segmentM = mask_clothes + mask_dress + mask_jacket

            skin_color = ger_average_color((mask_r_arm + mask_l_arm - mask_l_arm * mask_r_arm),
                                           (mask_r_arm + mask_l_arm - mask_l_arm * mask_r_arm) * input_P1)

            t_mask_clothes = (in_label2 == 5).float()
            t_mask_dress = (in_label2 == 6).float()
            t_mask_jacket = (in_label2 == 7).float()
            segment = t_mask_clothes + t_mask_dress + t_mask_jacket

            left_gap = data['Tleft'].float().cuda()
            right_gap = data['Tright'].float().cuda()
            boxes = data['Boxes'].float().cuda()
            in_complete_coor = data['complete_coor'].cuda()

            mask_fore = (in_label > 0).float()
            img_hole_hand = input_P1 * (1 - mask_r_arm) * (1 - mask_l_arm) * (1 - segmentM)
            invisible_torso = skin_color * (mask_r_arm + mask_l_arm + segmentM)
            img_hole_hand += invisible_torso
            sil = torch.zeros((in_label2.shape)).float().to(device)
            for b in range(sil.shape[0]):
                w = sil.shape[3]
                sil[b][:, :, int(right_gap[b].item()):w - int(left_gap[b].item())] = 1

            img_fore2 = input_P2 * (1 - segment)

            appearance = torch.cat([img_hole_hand, mask_fore, in_complete_coor], 1)
            ############ Optimize Discriminator ############
            requires_grad(generator, False)
            requires_grad(discriminator, True)

            fake_img, _ = generator(appearance=appearance, target_dense=target_dense,
                                    segment=segment)
            fake_img *= sil
            fake_img *= (1 - segment)
            fake_pred = discriminator(fake_img, pose=target_dense)
            real_pred = discriminator(img_fore2, pose=target_dense)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            discriminator.zero_grad()
            d_loss.backward()
            d_optim.step()

            d_regularize = i % args.d_reg_every == 0
            if d_regularize:
                img_fore2.requires_grad = True
                real_pred = discriminator(img_fore2, pose=target_dense)
                r1_loss = d_r1_loss(real_pred, img_fore2)
                discriminator.zero_grad()
                (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
                d_optim.step()

            ############## Optimize Generator ##############
            requires_grad(generator, True)
            requires_grad(discriminator, False)
            fake_img, _ = generator(appearance=appearance, target_dense=target_dense,
                                    segment=segment)
            fake_img *= sil
            fake_img *= (1 - segment)
            fake_pred = discriminator(fake_img, pose=target_dense)
            g_loss = g_nonsaturating_loss(fake_pred)

            #loss_dict["g"] = g_loss

            # reconstruction loss: L1 and VGG loss + face identity loss
            g_loss += criterionL1(fake_img, img_fore2)
            g_loss += criterionVGG(fake_img, img_fore2)

            processed_real_face = getFace(input_P2, boxes, left_gap, right_gap)
            if (processed_real_face is not None):
                features_real_face = sphereface_net(processed_real_face)
                processed_fake_face = getFace(fake_img, boxes, left_gap, right_gap)
                features_fake_face = sphereface_net(processed_fake_face)
                g_cos = 1. - criterionCOS(features_real_face, features_fake_face).mean()
                g_loss += g_cos

            generator.zero_grad()
            g_loss.backward()
            g_optim.step()

            ############ Optimization Done ############
            accumulate(g_ema, g_module, accum)

            if rank_number == 0:
                if i % 100 == 0:
                    print('Epoch: [{0}/{1}] Iter: [{2}/{3}]\t'.format(epoch, args.epoch, i, len(loader))
                           +
                           f"d: {d_loss:.4f}; g: {g_loss:.4f}; face: {g_cos:.4f};"
                       )
                    writer.add_scalar('generator loss', g_loss, step)
                    writer.add_scalar('face loss', g_cos, step)
                    writer.add_scalar('discriminator loss', d_loss, step)

                if i % 5000 == 0:
                    with torch.no_grad():
                        g_ema.eval()
                        sample, _ = g_ema(appearance=appearance[:args.n_sample],
                                          target_dense=target_dense[:args.n_sample],
                                          segment=segment[:args.n_sample])
                    rgb_uv = torch.nn.functional.grid_sample(img_hole_hand, in_complete_coor.permute(0, 2, 3, 1))
                    sample *= sil
                    sample *= (1 - segment)
                    writer.add_image('generated', torchvision.utils.make_grid(sample), step)
                    writer.add_image('gt', torchvision.utils.make_grid(img_fore2), step)
                    writer.add_image('input', torchvision.utils.make_grid(rgb_uv), step)
                    writer.add_image('input_image', torchvision.utils.make_grid(img_hole_hand), step)
                    writer.add_image('face_gt', torchvision.utils.make_grid(processed_real_face), step)
                    writer.add_image('face_fake', torchvision.utils.make_grid(processed_fake_face), step)
        ###################################
        ############ END EPOCH ############
        ###################################
        if rank_number == 0:
            torch.save(
                {
                    "g": g_module.state_dict(),
                    "d": d_module.state_dict(),
                    "g_ema": g_ema.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                }, 'checkpoint_pose_transfer' + '/pose_transfer_' + str(epoch) + '.pth')



if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser(description="Pose with Style trainer")
    parser.add_argument("--epoch", type=int, default=10, help="total training epochs")
    parser.add_argument("--batch", type=int, default=1, help="batch sizes for each gpus")
    parser.add_argument("--workers", type=int, default=2, help="batch sizes for each gpus")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--ckpt", type=str, default='pretrained/posewithstyle.pt', help="pretrained checkpoint dir")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--faceloss", action="store_true", help="add face loss when faces are detected")
    parser.add_argument("--pairLst", type=str, default='custom_DeepFashion/face-only-pairs.csv')
    parser.add_argument("--dataroot", type=str, default='custom_DeepFashion')
    parser.add_argument("--phase", type=str, default='train')
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(args.local_rank)
    synchronize()

    if get_rank() == 0:
        if not os.path.exists('checkpoint_pose_transfer'):
            os.makedirs('checkpoint_pose_transfer')
        writer = SummaryWriter('runs/pose_transfer')

    args.latent = 2048
    args.n_mlp = 8
    args.start_epoch = 0

    generator = GeneratorCustom(512, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    discriminator = Discriminator(512, channel_multiplier=args.channel_multiplier).to(device)
    g_ema = GeneratorCustom(512, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    sphereface_net = getattr(sphereface, 'sphere20a')()
    sphereface_net.load_state_dict(torch.load('pretrained/sphere20a_20171020.pth'))
    sphereface_net.to(device)
    sphereface_net.eval()
    sphereface_net.feature = True

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
    )
    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

    generator.load_state_dict(ckpt["g"])
    discriminator.load_state_dict(ckpt["d"])
    g_ema.load_state_dict(ckpt["g_ema"])

    g_optim.load_state_dict(ckpt["g_optim"])
    d_optim.load_state_dict(ckpt["d_optim"])

    generator = nn.parallel.DistributedDataParallel(
        generator,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False
    )

    discriminator = nn.parallel.DistributedDataParallel(discriminator,
                                                        device_ids=[args.local_rank],
                                                        output_device=args.local_rank,
                                                        broadcast_buffers=False)
    dataset = CustomDeepFashionDatasetHD()
    dataset.initialize(args, 512)
    sampler = data_sampler(dataset, shuffle=True, distributed=True)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
        num_workers=args.workers,
        shuffle=False,
    )
    train(args, loader, sampler, generator, discriminator, g_optim, d_optim, g_ema, device)