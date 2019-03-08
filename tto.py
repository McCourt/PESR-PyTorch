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
        elif arg == '--scale':
            scale = int(opt)
        else:
            raise UserWarning('Redundant argument {}'.format(arg))

    try:
        params = load_parameters()
        print('Parameters loaded')
        print(''.join(['-' for i in range(30)]))
        for i in params['tto']:
            print('{:<15s} -> {}'.format(str(i), params['tto'][i]))
        device_name = params['tto']['device_name']
        num_epoch = params['tto']['num_epoch']
        beta = params['tto']['beta']
        beta_1 = params['tto']['beta_1']
        learning_rate = params['tto']['learning_rate']
        save = params['tto']['save']
        rgb_shuffle = params['tto']['rgb_shuffle']
        print_every = params['tto']['print_every']
        method = params['tto']['method']
    except Exception as e:
        print(e)
        raise ValueError('Parameter not found.')

    out_dir = os.path.join(params['common']['root_dir'], params['common']['tto_dir'].format(model, method, dataset, scale))
    lr_dir = os.path.join(params['common']['root_dir'], params['common']['lr_dir'].format(dataset, scale))
    hr_dir = os.path.join(params['common']['root_dir'], params['common']['hr_dir'].format(dataset))
    sr_dir = os.path.join(params['common']['root_dir'], params['common']['sr_dir'].format(model, dataset, scale))
    log_dir = os.path.join(params['common']['root_dir'], params['common']['tto_log'].format(model, method, dataset, scale))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(out_dir, 'sr')):
        os.makedirs(os.path.join(out_dir, 'sr'))
    if not os.path.exists(os.path.join(out_dir, 'dsr')):
        os.makedirs(os.path.join(out_dir, 'dsr'))

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

    down_sampler = BicubicDownSample()
    down_sampler = down_sampler.to(device)
    lr_loss = nn.MSELoss()
    l2_loss = nn.MSELoss()
    hr_loss = nn.MSELoss()

    discriminator = Discriminator_VGG_128()
    ckpt = torch.load(os.path.join(params['common']['root_dir'], params['tto']['disk_ckpt']))
    discriminator.load_state_dict(ckpt)
    discriminator.require_grad = False
    discriminator = discriminator.to(device)

    print('Begin TTO on device {}'.format(device))
    with open(os.path.join(log_dir), 'w') as f:
        title_formatter = '{:^5s} | {:^10s} | {:^10s} | {:^10s} | {:^10s} | {:^10s} | {:^10s} | {:^10s} | {:^10s}'
        title = title_formatter.format('Epoch', 'IMG Name', 'DS Loss', 'REG Loss', 'DIS Loss',
                                       'SR Loss', 'LR PSNR', 'SR PSNR', 'Runtime')
        print(''.join(['-' for i in range(110)]))
        print(title)
        print(''.join(['-' for i in range(110)]))
        f.write(title + '\n')
        report_formatter = '{:^5d} | {:^10s} | {:^10.4f} | {:^10.4f} | {:^10.2f} | {:^10.4f} | {:^10.4f} | {:^10.4f} ' \
                           '| {:^10.4f} '

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
            pad_h, pad_w = 128 - sr_img.shape[2] % 128, 128 - sr_img.shape[3] % 128
            crop_h, crop_w = sr_img.shape[2] // 128 * 128, sr_img.shape[3] // 128 * 128
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
                ds_in_tensor = down_sampler(sr_tensor)

                lr_l = lr_loss(ds_in_tensor, lr_tensor)
                l2_l = l2_loss(sr_tensor, org_tensor)
                vs_l = torch.mean(
                    -discriminator(
                        F.pad(sr_tensor, (0, pad_w, 0, pad_h), 'constant').permute(0, 2, 3, 1).view((-1, 128, 128, 3)).permute(0, 3, 1, 2)
                    )
                )

                l0_l = hr_loss(sr_tensor, hr_tensor)

                l = lr_l + beta * l2_l + beta_1 * vs_l
                l.backward()
                optimizer.step()
                scheduler.step()
                diff = time() - begin_time
                report = report_formatter.format(epoch, img_name, lr_l, l2_l, vs_l, l0_l, psnr(lr_l), psnr(l0_l), diff)
                if epoch % print_every == 0 or epoch == num_epoch - 1:
                    print(report)
                f.write(report + '\n')
            print(''.join(['-' for i in range(110)]))

            if save:
                sr_img = torch.clamp(torch.round(sr_tensor), 0., 255.).detach().cpu().numpy().astype(np.uint8)
                sr_img = np.moveaxis(sr_img, 1, -1).reshape((h, w, c)).astype(np.uint8)
                imwrite(os.path.join(out_dir, 'sr', img_name), sr_img, format='png', compress_level=0)
