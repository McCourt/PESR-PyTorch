import torch
import torch.nn as nn
import torch.nn.functional as F
from imageio import imwrite
import numpy as np
from time import time
import os, sys, getopt

from model.discriminator.Discriminator_VGG import Discriminator_VGG_128
from model.downscaler.bicubic import BicubicDownSample
# from downscaler.conv import ConvolutionDownscale
from src.helper import psnr, load_parameters, ChannelGradientShuffle
from src.dataset import SRTTODataset
from loss.shiftloss import ShiftLoss
from loss.discloss import GanLoss
from loss.regloss import RegularizationLoss
from loss.dsloss import DownScaleLoss
from loss.psnr import PSNR

if __name__ == '__main__':
    args = sys.argv[1:]
    long_opts = ['model-name=', 'dataset=', 'scale=']
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
        elif arg == '--scale':
            scale = int(opt)
        else:
            raise UserWarning('Redundant argument {}'.format(arg))

    try:
        params = load_parameters()
        print('Parameters loaded')
        print(''.join(['-' for i in range(30)]))
        common, tto_params = params['common'], params['tto']
        for i in sorted(tto_params):
            print('{:<15s} -> {}'.format(str(i), tto_params[i]))
        device_name = tto_params['device_id']
        num_epoch = tto_params['num_epoch']
        beta = tto_params['beta']
        beta_1 = tto_params['beta_1']
        beta_2 = tto_params['beta_2']
        learning_rate = tto_params['learning_rate']
        save = tto_params['save']
        rgb_shuffle = tto_params['rgb_shuffle']
        print_every = tto_params['print_every']
        method = tto_params['method']
    except Exception as e:
        print(e)
        raise ValueError('Parameter not found.')

    root_dir = common['root_dir']
    lr_dir = os.path.join(root_dir, common['s0_dir'], tto_params['lr_dir'])
    hr_dir = os.path.join(root_dir, common['s0_dir'], tto_params['hr_dir'])
    sr_dir = os.path.join(root_dir, common['s1_dir'], tto_params['sr_dir'])
    out_dir = os.path.join(root_dir, common['s2_dir'], tto_params['tto_dir'])
    osr_dir, dsr_dir = os.path.join(out_dir, 'sr'), os.path.join(out_dir, 'dsr')
    log_dir = os.path.join(out_dir, 'tto.log')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(osr_dir):
        os.makedirs(osr_dir)
    if not os.path.exists(dsr_dir):
        os.makedirs(dsr_dir)

    """
    down_sampler = ConvolutionDownscale()
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
    device = torch.device(device_name if torch.cuda.is_available else 'cpu')
    dataset = SRTTODataset(hr_dir, lr_dir, sr_dir)

    shift_loss = ShiftLoss().to(device)
    gan_loss = GanLoss().to(device)
    ds_loss = DownScaleLoss().to(device)
    reg_loss = RegularizationLoss().to(device)
    hr_psnr = PSNR()

    print('Begin TTO on device {}'.format(device))
    with open(os.path.join(log_dir), 'w') as f:
        title_formatter = '{:^5s} | {:^10s} | {:^10s} | {:^10s} | {:^10s} | {:^10s} | {:^10s} | {:^10s} | {:^10s}'
        title = title_formatter.format('Epoch', 'IMG Name', 'DS Loss', 'REG Loss', 'DIS Loss',
                                       'SHI Loss', 'LR PSNR', 'SR PSNR', 'Runtime')
        splitter = ''.join(['-' for i in range(len(title))])
        print(splitter)
        print(title)
        print(splitter)
        f.write(title + '\n')
        report_formatter = '{:^5d} | {:^10s} | {:^10.4f} | {:^10.4f} | {:^10.4f} | {:^10.4f} | {:^10.4f} | {:^10.4f} | {:^10.4f} '

        for img_dic in dataset:
            img_name = img_dic['name']
            lr_img = img_dic['lr']
            sr_img = img_dic['sr']
            hr_img = img_dic['hr']

            lr_img = np.expand_dims(lr_img, axis=0)
            sr_img = np.expand_dims(sr_img, axis=0)
            hr_img = np.expand_dims(hr_img, axis=0)
            _, c, h, w = sr_img.shape

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
            channel = [0, 1, 2]
            for epoch in range(num_epoch):
                print(title, end='\r')
                begin_time = time()
                optimizer.zero_grad()

                ds_l = ds_loss(sr=sr_tensor, lr=lr_tensor)
                reg_l = reg_loss(sr=org_tensor, sr_tto=sr_tensor)
                vs_l = gan_loss(sr_tensor)
                sh_l = shift_loss(sr_tensor, lr_tensor)
                hr_p = hr_psnr(sr_tensor, hr_tensor)
                ds_p = psnr(ds_l)

                l = ds_l + beta * reg_l + beta_1 * vs_l + beta_2 * sh_l
                l.backward()
                optimizer.step()
                scheduler.step()
                diff = time() - begin_time
                report = report_formatter.format(epoch, img_name, ds_l, reg_l, vs_l, sh_l, ds_p, hr_p, diff)
                if epoch % print_every == 0 or epoch == num_epoch - 1:
                    print(report)
                f.write(report + '\n')
            print(splitter)

            if save:
                sr_img = torch.clamp(torch.round(sr_tensor), 0., 255.).detach().cpu().numpy().astype(np.uint8)
                sr_img = np.moveaxis(sr_img, 1, -1).reshape((h, w, c)).astype(np.uint8)
                imwrite(os.path.join(osr_dir, img_name), sr_img, format='png', compress_level=0)
