from downsample.conv import ConvolutionDownscale
import torch
from torch import nn
from imageio import imread, imwrite
import numpy as np
import os
from time import time
from skimage.color import gray2rgb
from helper import load_checkpoint, psnr

model = 'EnhanceNet-E'
dataset = 'Urban100'
method = 'convolution_down_sample'
scale = 4
device_name = 'cuda:2'
num_epoch = 50
beta = 1
learning_rate = 5e-2
save = True

root_dir = '/usr/xtmp/superresoluter/superresolution'
out_dir = os.path.join(root_dir, 'imgs/stage_two_image/{}-{}/{}/x{}'.format(model, method, dataset, scale))
lr_dir = os.path.join(root_dir, 'imgs/source_image/{}/LR_PIL/x{}'.format(dataset, scale))
hr_dir = os.path.join(root_dir, 'imgs/source_image/{}/HR'.format(dataset))
sr_dir = os.path.join(root_dir, 'imgs/stage_one_image/{}/{}/x{}/'.format(model, dataset, scale))
log_dir = os.path.join(root_dir, 'imgs/stage_two_image/{}-{}/{}/x{}'.format(model, method, dataset, scale))
device = torch.device(device_name if torch.cuda.is_available else 'cpu')

if __name__ == '__main__':
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(out_dir, 'sr')):
        os.makedirs(os.path.join(out_dir, 'sr'))
    if not os.path.exists(os.path.join(out_dir, 'dsr')):
        os.makedirs(os.path.join(out_dir, 'dsr'))
    print(device)
    bds = ConvolutionDownscale()
    ckpt = load_checkpoint(load_dir='./checkpoints/', map_location=None, model_name='down_sample')
    try:
        if ckpt is not None:
            print('recovering from checkpoints...')
            bds.load_state_dict(ckpt['model'], map_location=device)
            print('resuming training')
    except Exception as e:
        print(e)
        raise FileNotFoundError('Check checkpoints')
    bds = bds.to(device)
    lr_loss = nn.MSELoss(reduction='mean')
    l2_loss = nn.MSELoss(reduction='mean')
    hr_loss = nn.MSELoss(reduction='mean')
    with open(os.path.join(log_dir, '{}.log'.format(model)), 'w') as f:
        for img_name in os.listdir(hr_dir):
            lr_img = np.array(imread(os.path.join(lr_dir, img_name)))
            sr_img = np.array(imread(os.path.join(sr_dir, img_name)))
            hr_img = np.array(imread(os.path.join(hr_dir, img_name)))

            if len(lr_img.shape) == 2:
                lr_img = gray2rgb(lr_img)
            if len(sr_img.shape) == 2:
                sr_img = gray2rgb(sr_img)
            if len(hr_img.shape) == 2:
                hr_img = gray2rgb(hr_img)

            lr_img = np.expand_dims(np.moveaxis(lr_img, -1, 0).astype(np.float32), axis=0)
            sr_img = np.expand_dims(np.moveaxis(sr_img, -1, 0).astype(np.float32), axis=0)
            hr_img = np.expand_dims(np.moveaxis(hr_img, -1, 0).astype(np.float32), axis=0)
            _, c, h, w = sr_img.shape

            lr_tensor = torch.from_numpy(lr_img).type('torch.cuda.FloatTensor').to(device)
            in_tensor = torch.from_numpy(sr_img).type('torch.cuda.FloatTensor').to(device)
            org_tensor = torch.from_numpy(sr_img).type('torch.cuda.FloatTensor').to(device)
            hr_tensor = torch.from_numpy(hr_img).type('torch.cuda.FloatTensor').to(device)
            in_tensor.requires_grad = True
            optimizer = torch.optim.Adam([in_tensor], lr=learning_rate)
            psnrs = []
            begin_time = time()
            for epoch in range(num_epoch):
                ds_in_tensor = bds(in_tensor)
                lr_l = lr_loss(ds_in_tensor, lr_tensor)
                l2_l = l2_loss(in_tensor, org_tensor)
                l0_l = hr_loss(in_tensor, hr_tensor)
                l = lr_l + beta * l2_l
                l.backward()
                optimizer.step()
                in_tensor.requires_grad = True
                report = '{} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}'.format(img_name, lr_l, l2_l, l0_l,psnr(lr_l), psnr(lr_l), psnr(l0_l))
                if epoch % 10 == 0 or epoch == num_epoch - 1:
                    print(report)
                f.write(report)

            if save:
                sr_img = torch.clamp(torch.round(in_tensor), 0., 255.).detach().cpu().numpy().astype(np.uint8)
                sr_img = np.moveaxis(sr_img, 1, -1).reshape((h, w, c)).astype(np.uint8)
                imwrite(os.path.join(out_dir, 'sr', img_name), sr_img, format='png', compress_level=0)
        print(np.mean(psnrs))
