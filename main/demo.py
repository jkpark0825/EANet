import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import time
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, process_bbox, augmentation
from utils.vis import save_obj, vis_mesh
from utils.human_models import mano
import glob
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', type=str, default=29, dest='test_epoch')
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--input', type=str, default='example_image1.png', dest='input')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True
model = get_model('test')
model = DataParallel(model).cuda()
model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % int(args.test_epoch))
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()
img = load_img(args.input)
height, width = img.shape[:2]
bbox = [0, 0, width, height] 
bbox = process_bbox(bbox, width, height)
img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, 'test')
img = transform(img.astype(np.float32))/255.
img = img.cuda()[None,:,:,:]
inputs = {'img': img}
targets = {}
meta_info = {}
with torch.no_grad():
    out = model(inputs, targets, meta_info, 'test')
    
img = (img[0].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)  
rmano_mesh = out['rmano_mesh_cam'][0].cpu().numpy()
lmano_mesh = out['lmano_mesh_cam'][0].cpu().numpy()
rel_trans = out['rel_trans'][0].cpu().numpy()

save_obj(rmano_mesh*np.array([1,-1,-1]), mano.face['right'], 'demo_right.obj')
save_obj((lmano_mesh+rel_trans)*np.array([1,-1,-1]), mano.face['left'], 'demo_left.obj')
    