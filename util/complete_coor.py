import os
from PIL import Image
import numpy as np
import torch
from coordinate_completion_model import define_G as define_CCM
import threading

def pad_PIL(pil_img, top, right, bottom, left, color=(0, 0, 0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype(np.uint8))
    image_pil.save(image_path)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='data', help="path to DeepFashion dataset")
parser.add_argument('--coordinates_path', type=str, default='coord', help="path to partial coordinates dataset")
parser.add_argument('--phase', type=str, default='train', help="train or test")
parser.add_argument('--save_path', type=str, default='complete', help="path to save results")
parser.add_argument('--pretrained_model', type=str, default='CCM_epoch50.pt', help="path to save results")

args = parser.parse_args()

phase = args.phase

if not os.path.exists(os.path.join(args.save_path, phase)):
    os.makedirs(os.path.join(args.save_path, phase))

coor_completion_generator = define_CCM()
CCM_checkpoint = torch.load(args.pretrained_model, map_location=torch.device('cpu'))
coor_completion_generator.load_state_dict(CCM_checkpoint["g"])
coor_completion_generator.eval()
for param in coor_completion_generator.parameters():
    coor_completion_generator.requires_grad = False

class App(threading.Thread):
    def __init__(self, chunks):
        threading.Thread.__init__(self)
        self.file_path = chunks

    def run(self):
        for i in range(len(self.file_path)):
            #print ('%d/%d'%(i, len(images)))
            im_name = self.file_path[i].split(os.sep)[-1]
            # get image
            #path = os.path.join(args.dataroot, phase, im_name)
            path = os.path.join(args.dataroot, phase, 'candidate', im_name)
            im = Image.open(path)
            w, h = im.size

            # get uv coordinates
            uvcoor_root = os.path.join(args.dataroot, phase, args.coordinates_path)
            uv_coor_path = os.path.join(uvcoor_root, im_name.split('.')[0]+'_iuv_uv_coor.npy')
            uv_mask_path = os.path.join(uvcoor_root, im_name.split('.')[0]+'_iuv_uv_mask.png')
            #uv_symm_mask_path = os.path.join(uvcoor_root, im_name.split('.')[0]+'_iuv_uv_symm_mask.png')
            if (os.path.exists(uv_coor_path)):
                # read high-resolution coordinates
                uv_coor = np.load(uv_coor_path)
                uv_mask = np.array(Image.open(uv_mask_path))/255
                #uv_symm_mask = np.array(Image.open(uv_symm_mask_path))/255

                # uv coor
                shift = int((h-w)/2)
                #uv_coor[:,:,0] = uv_coor[:,:,0] + shift # put in center
                #uv_coor = ((2*uv_coor/(h-1))-1)
                #uv_coor = uv_coor*np.expand_dims(uv_mask,2) + (-10*(1-np.expand_dims(uv_mask,2)))

                x1 = shift
                x2 = h-(w+x1)
                im = pad_PIL(im, 0, x2, 0, x1, color=(0, 0, 0))

                ## coordinate completion
                uv_coor_pytorch = torch.from_numpy(uv_coor).float().permute(2, 0, 1).unsqueeze(0) # from h,w,c to 1,c,h,w
                uv_mask_pytorch = torch.from_numpy(uv_mask).unsqueeze(0).unsqueeze(0).float() #1xchw
                with torch.no_grad():
                    coor_completion_generator.eval()
                    complete_coor = coor_completion_generator(uv_coor_pytorch, uv_mask_pytorch)
                uv_coor = complete_coor[0].permute(1,2,0).data.cpu().numpy()
                #uv_confidence = np.stack([uv_mask-uv_symm_mask, uv_symm_mask, 1-uv_mask], 2)

                im = torch.from_numpy(np.array(im)).permute(2, 0, 1).unsqueeze(0).float()
                rgb_uv = torch.nn.functional.grid_sample(im, complete_coor.permute(0,2,3,1))
                rgb_uv = rgb_uv[0].permute(1,2,0).data.cpu().numpy()

                # saving
                #save_image(rgb_uv, os.path.join(args.save_path, phase, im_name.split('.jpg')[0]+'.png'))
                np.save(os.path.join(args.dataroot, phase, args.save_path,'%s_uv_coor.npy'%(im_name.split('.')[0])), uv_coor)
                #save_image(uv_confidence*255, os.path.join(args.save_path, phase, '%s_conf.png'%(im_name.split('.')[0])))


def chunks(lst:list, n:int) -> list:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_abspath_files_in_directory(directory:str) -> list:
    return [os.path.abspath(os.path.join(directory,path)) for path in  os.listdir(directory)]

if __name__ == "__main__":
    file_paths = get_abspath_files_in_directory("data/train/candidate")
    file_paths = chunks(file_paths, 400)

    for index, chunk in enumerate(file_paths):
       App(chunk).start()