import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.helper import Timer, load_parameters
from dataset import SRTrainDataset, SRTestDataset
from loss import PSNR, DownScaleLoss
from model import Model
import os, sys
import getopt
import numpy as np
from imageio import imwrite

if __name__ == '__main__':
    print('{} GPUs Available'.format(torch.cuda.device_count()))

    # Load system arguments
    args = sys.argv[1:]
    long_opts = ['mode=']

    try:
        optlist, args = getopt.getopt(args, '', long_opts)
    except getopt.GetoptError as e:
        print(str(e))
        sys.exit(2)

    for arg, opt in optlist:
        if arg == '--mode':
            mode = str(opt)
            if mode not in ['train', 'test']:
                raise Exception('Wrong pipeline mode detected')
        else:
            raise Exception("Redundant Argument Detected")

    # Load JSON arguments
    try:
        params = load_parameters(path='parameter.json')
        print('Parameters loaded')
        print(''.join(['-' for i in range(30)]))
        train_params, test_params, common_params = params['train'], params['test'], params['common']
        for i in sorted(train_params):
            print('{:<15s} -> {}'.format(str(i), train_params[i]))
    except Exception as e:
        print(e)
        raise ValueError('Parameter not found.')

    # Prepare common parameters
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    up_sampler, down_sampler = common_params['up_sampler'], common_params['down_sampler']
    if up_sampler is None and down_sampler is None:
        raise Exception('You must define either an upscale model or a downscale model for super resolution')

    # Prepare all directory and devices
    root_dir = common_params['root_dir']
    is_train = True if mode == 'train' else False
    begin_epoch = train_params['begin_epoch'] if is_train else 0
    model_name = '+'.join([str(i) for i in [up_sampler, down_sampler]])
    trim, scale, psnr = common_params['trim'], common_params['scale'], PSNR()
    train_hr_dir = os.path.join(root_dir, common_params['s0_dir'], train_params['hr_dir'])
    train_lr_dir = os.path.join(root_dir, common_params['s0_dir'], train_params['lr_dir'])
    val_hr_dir = os.path.join(root_dir, common_params['s0_dir'], test_params['hr_dir'])
    val_lr_dir = os.path.join(root_dir, common_params['s0_dir'], test_params['lr_dir'])
    sr_dir = os.path.join(root_dir, common_params['s1_dir'], up_sampler, train_params['sr_dir'])
    if not os.path.isdir(sr_dir):
        os.makedirs(sr_dir)
    log_dir = os.path.join(root_dir, common_params['log_dir'].format(model_name))
    num_epoch = train_params['num_epoch'] if is_train else 1

    sr_ckpt = os.path.join(root_dir, common_params['ckpt_dir'].format(up_sampler))
    sr_model = Model(name=up_sampler, mode='upscaler', checkpoint=sr_ckpt, train=is_train, log=log_dir)
    sr_loss = DownScaleLoss().cuda() # nn.L1Loss().cuda()
    # ds_loss = DownScaleLoss(clip_round=False).cuda()

    # Define optimizer, learning rate scheduler, data source and data loader
    if is_train:
        lr = train_params['learning_rate'] * train_params['decay_rate'] ** begin_epoch
        if up_sampler is not None:
            sr_optimizer = torch.optim.Adam(sr_model.parameters(), lr=lr)
            sr_scheduler = torch.optim.lr_scheduler.ExponentialLR(sr_optimizer,
                                                                  gamma=train_params['decay_rate'],
                                                                  last_epoch=begin_epoch - 1)
        else:
            raise Exception('No trainable parameters in training mode')

        train_dataset = SRTrainDataset(
            hr_dir=train_hr_dir,
            lr_dir=train_lr_dir,
            h=train_params['window'][0],
            w=train_params['window'][1],
            scale=scale,
            num_per=train_params['num_per']
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_params['batch_size'],
            shuffle=True,
            num_workers=train_params['num_worker']
        )
    val_dataset = SRTestDataset(hr_dir=val_hr_dir, lr_dir=val_lr_dir)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=train_params['num_worker']
    )

    # Training loop and saver as checkpoints
    print('Using device {}'.format(device))
    print(sr_model.splitter)
    print(sr_model.t)
    print(sr_model.splitter)
    best_val = None
    for epoch in range(begin_epoch, num_epoch):
        if is_train:
            sr_model.train_step(train_loader, sr_optimizer, sr_scheduler, sr_loss)
        val_l = sr_model.test_step(val_loader, sr_loss)
        if best_val is None or best_val > val_l:
            if is_train:
                sr_model.save_checkpoint()
            best_val = val_l
            print('Saving best-by-far model at {}'.format(best_val))
