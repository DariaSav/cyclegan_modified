import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import argparse
import torch
import torchvision.transforms as transforms

from PIL import Image
import numpy as np

def change_domain(model, opt, img, domain):
    
    osize = [opt.fineSizex, opt.fineSizey]
    tr =  transforms.Compose([transforms.Resize(osize, Image.BICUBIC), transforms.ToTensor()])

    img = tr(img).unsqueeze(0).to(model.device)
    
    if domain == 'A':
            img = model.netG_B(img)
    elif domain == 'B':
            img = model.netG_A(img)

    img = img.data[0].cpu().float().numpy()

    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    img = img.astype(np.uint8)

    return Image.fromarray(img)

if __name__ == '__main__':

	opt = argparse.Namespace( aspect_ratio=1.0, 
		           batch_size=1, 
		           checkpoints_dir='./checkpoints', 
		           dataroot='/main/uni/diplom/pytorch-CycleGAN-and-pix2pix/datasets/day2night_clean/', 
		           dataset_mode='single', 
		           direction='AtoB', 
		           display_winsize=256, 
		           display_id = -1,
		           epoch='latest', 
		           eval=False, 
		           fineSizex=180,
		           fineSizey=320,
		           gpu_ids=[2], 
		           init_gain=0.02, 
		           init_type='normal', 
		           input_nc=3, 
		           isTrain=False, 
		           loadSize=256, 
		           max_dataset_size=float("inf"), 
		           model='cycle_gan', 
		           model_suffix='', 
		           n_layers_D=3, 
		           name='day2night_320_clean', 
		           ndf=64, 
		           netD='basic', 
		           netG='resnet_9blocks', 
		           ngf=64, no_dropout=True, 
		           no_flip=True, 
		           norm='instance', 
		           ntest=float("inf"), 
		           num_test=50, 
		           num_threads=1, 
		           output_nc=3, 
		           phase='test', 
		           resize_or_crop='resize_and_crop', 
		           results_dir='./results/', 
		           serial_batches=True, 
		           suffix='', 
		           verbose=False)


	model = create_model(opt)
	model.setup(opt)

	img_path = '/main/uni/diplom/pytorch-CycleGAN-and-pix2pix/datasets/day2night_clean/testB/frame_0d08c1ee-8931-470c-b067-297b74fcb0ee_00000-1280_720.jpg'

	img_A = Image.open(img_path).convert('RGB')

	change_domain(model, opt, img_A, 'A')





