import os
from PIL import Image
import numpy as np
from scipy.interpolate import griddata
import cv2
import threading
import torch
from coordinate_completion_model import define_G as define_CCM


dp_uv_lookup_256_np = np.load('util/dp_uv_lookup_256.npy')

coor_completion_generator = define_CCM()
CCM_checkpoint = torch.load('../CCM_epoch50.pt', map_location=torch.device('cpu'))
coor_completion_generator.load_state_dict(CCM_checkpoint["g"])
coor_completion_generator.eval()
for param in coor_completion_generator.parameters():
    coor_completion_generator.requires_grad = False

def pad_PIL(pil_img, top, right, bottom, left, color=(0, 0, 0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def getSymXYcoordinates(iuv, resolution = 512):
    xy, xyMask = getXYcoor(iuv, resolution = resolution)
    f_xy, f_xyMask = getXYcoor(flip_iuv(np.copy(iuv)), resolution = resolution)
    f_xyMask = np.clip(f_xyMask-xyMask, a_min=0, a_max=1)
    # combine actual + symmetric
    combined_texture = xy*np.expand_dims(xyMask,2) + f_xy*np.expand_dims(f_xyMask,2)
    combined_mask = np.clip(xyMask+f_xyMask, a_min=0, a_max=1)
    return combined_texture, combined_mask, f_xyMask

def flip_iuv(iuv):
    POINT_LABEL_SYMMETRIES = [ 0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23]
    i = iuv[:,:,0]
    u = iuv[:,:,1]
    v = iuv[:,:,2]
    i_old = np.copy(i)
    for part in range(24):
        if (part + 1) in i_old:
            annot_indices_i = i_old == (part + 1)
            if POINT_LABEL_SYMMETRIES[part + 1] != part + 1:
                    i[annot_indices_i] = POINT_LABEL_SYMMETRIES[part + 1]
            if part == 22 or part == 23 or part == 2 or part == 3 : #head and hands IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!
                    u[annot_indices_i] = 255-u[annot_indices_i]
            if part == 0 or part == 1: # torso
                    v[annot_indices_i] = 255-v[annot_indices_i]
    return np.stack([i,u,v],2)

def getXYcoor(iuv, resolution = 512):
    x, y, u, v = mapper(iuv, resolution)
    # A meshgrid of pixel coordinates
    nx, ny = resolution, resolution
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
    ## get x,y coordinates
    uv_y = griddata((v, u), y, (Y, X), method='linear')
    uv_y_ = griddata((v, u), y, (Y, X), method='nearest')
    uv_y[np.isnan(uv_y)] = uv_y_[np.isnan(uv_y)]
    uv_x = griddata((v, u), x, (Y, X), method='linear')
    uv_x_ = griddata((v, u), x, (Y, X), method='nearest')
    uv_x[np.isnan(uv_x)] = uv_x_[np.isnan(uv_x)]
    # get mask
    uv_mask = np.zeros((ny,nx))
    uv_mask[np.ceil(v).astype(int),np.ceil(u).astype(int)]=1
    uv_mask[np.floor(v).astype(int),np.floor(u).astype(int)]=1
    uv_mask[np.ceil(v).astype(int),np.floor(u).astype(int)]=1
    uv_mask[np.floor(v).astype(int),np.ceil(u).astype(int)]=1
    kernel = np.ones((3,3),np.uint8)
    uv_mask_d = cv2.dilate(uv_mask,kernel,iterations = 1)
    # update
    coor_x = uv_x * uv_mask_d
    coor_y = uv_y * uv_mask_d
    coor_xy = np.stack([coor_x, coor_y], 2)
    return coor_xy, uv_mask_d

def mapper(iuv, resolution=512):
    H, W, _ = iuv.shape
    iuv_raw = iuv[iuv[:, :, 0] > 0]
    x = np.linspace(0, W-1, W).astype(np.int)
    y = np.linspace(0, H-1, H).astype(np.int)
    xx, yy = np.meshgrid(x, y)
    xx_rgb = xx[iuv[:, :, 0] > 0]
    yy_rgb = yy[iuv[:, :, 0] > 0]
    # modify i to start from 0... 0-23
    i = iuv_raw[:, 0] - 1
    u = iuv_raw[:, 1]
    v = iuv_raw[:, 2]
    uv_smpl = dp_uv_lookup_256_np[
    i.astype(np.int),
    v.astype(np.int),
    u.astype(np.int)
    ]
    u_f = uv_smpl[:, 0] * (resolution - 1)
    v_f = (1 - uv_smpl[:, 1]) * (resolution - 1)
    return xx_rgb, yy_rgb, u_f, v_f

class App(threading.Thread):
    def __init__(self, chunks):
        threading.Thread.__init__(self)
        self.file_path = chunks

    def run(self):
        class Args:
            save_path = 'data/train/coord/'
            dp_path = 'data/train/candidate_dense/'

        args = Args

        for i in range(len(self.file_path)):
            try:
                im_name = self.file_path[i].split(os.sep)[-1]
                dp = os.path.join(args.dp_path, im_name)

                iuv = np.array(Image.open(dp))
                h, w, _ = iuv.shape
                if np.sum(iuv[:,:,0]==0)==(h*w):
                    print ('no human: invalid image %d: %s'%(i, im_name))
                else:
                    uv_coor, uv_mask, uv_symm_mask  = getSymXYcoordinates(iuv, resolution = 512)
                    shift = int((h - w) / 2)
                    x1 = shift
                    x2 = h - (w + x1)
                    im = pad_PIL(im, 0, x2, 0, x1, color=(0, 0, 0))
                    uv_coor_pytorch = torch.from_numpy(uv_coor).float().permute(2, 0, 1).unsqueeze(
                        0)  # from h,w,c to 1,c,h,w
                    uv_mask_pytorch = torch.from_numpy(uv_mask).unsqueeze(0).unsqueeze(0).float()  # 1xchw
                    with torch.no_grad():
                        coor_completion_generator.eval()
                        complete_coor = coor_completion_generator(uv_coor_pytorch, uv_mask_pytorch)
                    uv_coor = complete_coor[0].permute(1, 2, 0).data.cpu().numpy()
                    np.save(os.path.join('data', 'train', 'coord', '%s_uv_coor.npy' % (im_name.split('.')[0])),
                        uv_coor)
            except:
                pass


def chunks(lst:list, n:int) -> list:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_abspath_files_in_directory(directory:str) -> list:
    return [os.path.abspath(os.path.join(directory,path)) for path in  os.listdir(directory)]

file_paths = get_abspath_files_in_directory("data/train/candidate_dense")
file_paths = chunks(file_paths, 400)

for index, chunk in enumerate(file_paths):
    App(chunk).start()