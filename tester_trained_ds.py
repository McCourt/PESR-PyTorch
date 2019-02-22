import random
from models.Discriminator_VGG import Discriminator_VGG_128
from downsample.conv import ConvolutionDownscale
from downsample.bicubic import BicubicDownSample
import torch
from torch import nn
import torch.nn.functional as F
from imageio import imread, imwrite
import numpy as np
import os, sys
from time import time
from skimage.color import gray2rgb
from helper import load_checkpoint, psnr, load_parameters, ChannelGradientShuffle
import getopt


if __name__ == '__main__':
    args = sys.argv[1:]
    long_opts = ['model-name=', 'dataset=', 'method=', 'scale=']
    try:
        optlist, args = getopt.getopt(args, '', long_opts)
    except getopt.GetoptError as e:
        print(str(e))
        sys.exit(2)

    for arg, opt in optlist:
        if arg == '--model-name':
            model = str(opt)
        elif arg == '--dataset':
            dataset = str(opt)
        elif arg == '--method':
            method = str(opt)
        elif arg == '--scale':
            scale = int(opt)
        else:
            raise UserWarning('Redundant argument {}'.format(arg))

    try:
        parameters = load_parameters(method)
        print(parameters)
        device_name = parameters['device_name']
        num_epoch = parameters['num_epoch']
        beta = parameters['beta']
        beta_1 = parameters['beta_1']
        learning_rate = parameters['learning_rate']
        save = parameters['save']
        rgb_shuffle = parameters['rgb_shuffle']
    except Exception as e:
        print(e)
        raise ValueError('Parameter not found.')

    root_dir = '/usr/xtmp/superresoluter/superresolution'
    out_dir = os.path.join(root_dir, 'imgs/stage_two_image/{}-{}/{}/x{}'.format(model, method, dataset, scale))
    lr_dir = os.path.join(root_dir, 'imgs/source_image/{}/LR_PIL/x{}'.format(dataset, scale))
    hr_dir = os.path.join(root_dir, 'imgs/source_image/{}/HR'.format(dataset))
    sr_dir = os.path.join(root_dir, 'imgs/stage_one_image/{}/{}/x{}/'.format(model, dataset, scale))
    log_dir = os.path.join(root_dir, 'imgs/stage_two_image/{}-{}/{}/x{}'.format(model, method, dataset, scale))
    device = torch.device(device_name if torch.cuda.is_available else 'cpu')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(out_dir, 'sr')):
        os.makedirs(os.path.join(out_dir, 'sr'))
    if not os.path.exists(os.path.join(out_dir, 'dsr')):
        os.makedirs(os.path.join(out_dir, 'dsr'))
    print(device)
    bds = BicubicDownSample() #ConvolutionDownscale()
    """
    ckpt = load_checkpoint(load_dir='./checkpoints/', map_location=None, model_name='down_sample')
    try:
        if ckpt is not None:
            print('recovering from checkpoints...')
            bds.load_state_dict(ckpt['model'])
            print('resuming training')
    except Exception as e:
        print(e)
        raise FileNotFoundError('Check checkpoints')
    """
    bds = bds.to(device)
    lr_loss = nn.MSELoss()
    l2_loss = nn.MSELoss()
    hr_loss = nn.MSELoss()
    vs_loss = Discriminator_VGG_128()
    ckpt = torch.load("./checkpoints/155000_D.pth")
    vs_loss.load_state_dict(ckpt)
    vs_loss.require_grad = False
    vs_loss = vs_loss.to(device)
    """
    ckpt = torch.load('')
    try:
        if ckpt is not None:
            print('recovering from checkpoints...')
            vs_loss.load_state_dict(ckpt['model'])
            print('resuming training')
    except Exception as e:
        print(e)
        raise FileNotFoundError('Check checkpoints')
    """
    with open(os.path.join(log_dir, '{}.log'.format(model)), 'w') as f:
        for img_name in sorted(os.listdir(hr_dir)):
            lr_img = np.array(imread(os.path.join(lr_dir, img_name)))
            sr_img = np.array(imread(os.path.join(sr_dir, img_name)))
            hr_img = np.array(imread(os.path.join(hr_dir, img_name)))

            if len(lr_img.shape) == 2:
                lr_img = gray2rgb(lr_img)
            if len(sr_img.shape) == 2:
                sr_img = gray2rgb(sr_img)
            if len(hr_img.shape) == 2:
                hr_img = gray2rgb(hr_img)
            '''
            h_0, w_0, c_0 = sr_img.shape
            pad_h, pad_w = h % 128, w % 128
            sr_img = np.pad(sr_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

            h, w, c = hr_img.shape
            pad_h, pad_w = 128 - h % 128, 128 - w % 128
            hr_img = np.pad(hr_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

            h, w, c = lr_img.shape
            pad_h, pad_w = 32 - h % 32, 32 - w % 32
            lr_img = np.pad(lr_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='contant')
            '''
            h, w, c = sr_img.shape
            lr_img = np.expand_dims(np.moveaxis(lr_img, -1, 0).astype(np.float32), 0)
            sr_img = np.expand_dims(np.moveaxis(sr_img, -1, 0).astype(np.float32), 0)
            hr_img = np.expand_dims(np.moveaxis(hr_img, -1, 0).astype(np.float32), 0)
            pad_h, pad_w = 128 - sr_img.shape[2] % 128, 128 - sr_img.shape[3] % 128

            lr_tensor = torch.from_numpy(lr_img).type('torch.cuda.FloatTensor').to(device)
            sr_tensor = torch.from_numpy(sr_img).type('torch.cuda.FloatTensor').to(device)
            org_tensor = torch.from_numpy(sr_img).type('torch.cuda.FloatTensor').to(device)
            hr_tensor = torch.from_numpy(hr_img).type('torch.cuda.FloatTensor').to(device)
            sr_tensor.requires_grad = True
            if rgb_shuffle:
                channel_shuffle = ChannelGradientShuffle.apply
                in_tensor = channel_shuffle(sr_tensor)
            else:
                in_tensor = sr_tensor
            optimizer = torch.optim.Adam([sr_tensor], lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            psnrs = []
            begin_time = time()
            channel = [0, 1, 2]
            for epoch in range(num_epoch):
                optimizer.zero_grad()
                ds_in_tensor = bds(sr_tensor)
                lr_l = lr_loss(ds_in_tensor, lr_tensor)
                l2_l = l2_loss(sr_tensor, org_tensor)
                vs_l = torch.sum(-vs_loss(F.pad(sr_tensor, (0, pad_w, 0, pad_h), 'constant').view((-1, 3, 128, 128))))
                l0_l = hr_loss(sr_tensor, hr_tensor)
                l = lr_l + beta * l2_l + beta_1 * vs_l
                l.backward()
                optimizer.step()
                scheduler.step()
                report = '{} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}'.format(img_name, lr_l, l2_l, l0_l,psnr(lr_l), psnr(lr_l), psnr(l0_l), vs_l)
                if epoch % 100 == 0 or epoch == num_epoch - 1:
                    print(report)
                f.write(report + '\n')

            if save:
                sr_img = torch.clamp(torch.round(sr_tensor), 0., 255.).detach().cpu().numpy().astype(np.uint8)
                sr_img = np.moveaxis(sr_img, 1, -1).reshape((h, w, c)).astype(np.uint8)
                imwrite(os.path.join(out_dir, 'sr', img_name), sr_img, format='png', compress_level=0)
