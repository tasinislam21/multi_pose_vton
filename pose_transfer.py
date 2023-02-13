import torch
import numpy as np
import torch.utils.data as data
from models import cStyleGAN
import os.path as osp
import torchvision.transforms as transforms
from PIL import Image
from util.coordinate_completion_model import define_G as define_CCM
from util.dp2coor import getSymXYcoordinates
import os

class Args:
    batchSize = 1
    datapairs = 'short.txt'
    dataroot = '../VTON-HD_512'
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

    def initialize(self, opt, size):
        self.opt = opt
        self.size = size
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.phase, 'image')
        self.dir_Label = os.path.join(opt.dataroot, opt.phase, 'image_label')
        self.dir_dense = os.path.join(opt.dataroot, opt.phase, 'densepose')

        self.transform = get_transform()
        self.transformBW = get_transform(normalize=False)

        self.human_names = []
        self.pose_names = []
        with open(os.path.join(opt.dataroot, opt.datapairs), 'r') as f:
            for line in f.readlines():
                h_name, c_name = line.strip().split()
                self.human_names.append(h_name)
                self.pose_names.append(c_name)

    def __getitem__(self, index):
        p_name = self.pose_names[index]
        h_name = self.human_names[index]

        candidate_path = osp.join(self.dir_P, h_name)
        label_path = osp.join(self.dir_Label, h_name+'.png')
        label_target_path = osp.join(self.dir_Label, p_name+'.png')
        source_dense_path = osp.join(self.dir_dense, h_name[:-4]+'_iuv.png')
        target_dense_path = osp.join(self.dir_dense, p_name[:-4]+'_iuv.png')

        P1_img = Image.open(candidate_path).convert('RGB')
        label_img = Image.open(label_path).convert('L')
        target_label_img = Image.open(label_target_path).convert('L')
        source_dense_img = np.array(Image.open(source_dense_path))
        target_dense_img = np.array(Image.open(target_dense_path))

        candidate = self.transform(P1_img)
        candidate = torch.nn.functional.pad(input=candidate, pad=(82, 82, 0, 0), mode='constant', value=0)
        label = self.transformBW(label_img) * 255
        label = torch.nn.functional.pad(input=label, pad=(82, 82, 0, 0), mode='constant', value=0)
        target_label = self.transformBW(target_label_img) * 255
        target_label = torch.nn.functional.pad(input=target_label, pad=(82, 82, 0, 0), mode='constant', value=0)
        target_dense = torch.from_numpy(target_dense_img).permute(2, 0, 1)
        target_dense = torch.nn.functional.pad(input=target_dense, pad=(82, 82, 0, 0), mode='constant', value=0)
        source_dense = torch.from_numpy(source_dense_img).permute(2, 0, 1)
        source_dense_short = torch.nn.functional.pad(input=source_dense, pad=(82, 82, 0, 0), mode='constant', value=0)
        return {'candidate': candidate, 'label': label, 'target_label': target_label,
                'source_dense': source_dense, 'target_dense': target_dense, 'source_dense_short': source_dense_short}

    def __len__(self):
            return len(self.human_names)

    def name(self):
        return 'KeyDataset'


# In[6]:


t = KeyDataset()
t.initialize(opt, 512)

dataloader = torch.utils.data.DataLoader(
            t,
            batch_size=opt.batchSize)

with torch.no_grad():
    G3 = cStyleGAN.GeneratorCustom(size = 512, style_dim = 2048, n_mlp = 8, channel_multiplier = 2).cuda()
    G3.load_state_dict(torch.load('checkpoint/60.pt'))
    G3.eval()

coor_completion_generator = define_CCM().cuda()
CCM_checkpoint = torch.load('checkpoint/CCM_epoch50.pt')
coor_completion_generator.load_state_dict(CCM_checkpoint["g"])
coor_completion_generator.eval()
for param in coor_completion_generator.parameters():
    coor_completion_generator.requires_grad = False

transform = get_transform()

def tensor2image(tensor):
    tensor = (tensor[0].clone() + 1) * 0.5 * 255
    tensor = tensor.cpu().clamp(0, 255)
    array = tensor.detach().numpy().astype('uint8')
    array = array.swapaxes(0, 1).swapaxes(1, 2)
    return array


for i, data in enumerate(dataloader):
    in_candidate = data['candidate'].cuda()
    in_label = data['label'].cuda()
    in_target_label = data['target_label'].cuda()
    in_source_dense = data['source_dense'][0].permute(1, 2, 0).numpy()
    in_target_dense = data['target_dense'].float().cuda()

    uv_coor, uv_mask, uv_symm_mask  = getSymXYcoordinates(in_source_dense, resolution = 512)

    mask_hair = (in_label == 2).float()
    mask_neck = (in_label == 10).float()
    mask_head = (in_label == 13).float()
    mask_touser = (in_label == 9).float()
    mask_r_leg = (in_label == 16).float()
    mask_l_leg = (in_label == 17).float()
    mask_r_shoe = (in_label == 18).float()
    mask_l_shoe = (in_label == 19).float()

    mask_clothes = (in_label == 5).float()
    mask_dress = (in_label == 6).float()
    mask_jacket = (in_label == 7).float()
    segmentM = mask_clothes + mask_dress + mask_jacket

    t_mask_clothes = (in_target_label == 5).float()
    t_mask_dress = (in_target_label == 6).float()
    t_mask_jacket = (in_target_label == 7).float()
    segment = t_mask_clothes + t_mask_dress + t_mask_jacket

    #img_hole_hand = in_candidate * (mask_hair + mask_neck + mask_head +
    #                                mask_touser + mask_r_leg +
    #                                mask_l_leg + mask_r_shoe + mask_l_shoe)
    img_hole_hand = in_candidate * (1 - segmentM)
    mask_fore = (in_label > 0).float()
    h, w = [512, 348]
    shift = int((h-w)/2) # center shift
    uv_coor[:,:,0] = uv_coor[:,:,0] + shift # put in center
    uv_coor = ((2*uv_coor/(h-1))-1)
    uv_coor = uv_coor*np.expand_dims(uv_mask,2) + (-10*(1-np.expand_dims(uv_mask,2)))

    # coordinate completion
    uv_coor_pytorch = torch.from_numpy(uv_coor).float().permute(2, 0, 1).unsqueeze(0) # from h,w,c to 1,c,h,w
    uv_mask_pytorch = torch.from_numpy(uv_mask).unsqueeze(0).unsqueeze(0).float() #1xchw
    with torch.no_grad():
        coor_completion_generator.eval()
        complete_coor = coor_completion_generator(uv_coor_pytorch.cuda(), uv_mask_pytorch.cuda())

    appearance = torch.cat([img_hole_hand, mask_fore, complete_coor], 1)

    fake_img, _ = G3(appearance=appearance, target_dense=in_target_dense,
                                        segment=segment)
    fake_img *= (1 - segment)

    img = Image.fromarray(tensor2image(fake_img))
    img.save('result/'+str(i)+'.jpg')

print("FINISHED")


