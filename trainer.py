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

    # Define upscale model and data parallel
    if up_sampler is not None:
        sr_ckpt = os.path.join(root_dir, common_params['ckpt_dir'].format(up_sampler))
        sr_model = Model(name=up_sampler, mode='upscaler', checkpoint=sr_ckpt, train=is_train)
        sr_loss = nn.L1Loss().cuda()

    # Define downscale model and data parallel and loss functions
    if down_sampler is not None:
        ds_ckpt = os.path.join(root_dir, common_params['ckpt_dir'].format(down_sampler))
        ds_model = Model(name=down_sampler, mode='downscaler', checkpoint=ds_ckpt, train=is_train)
        ds_loss = nn.MSELoss().cuda()
    else:
        ds_loss = DownScaleLoss(clip_round=False).cuda()

    # Define optimizer, learning rate scheduler, data source and data loader
    if is_train:
        lr = train_params['learning_rate'] * train_params['decay_rate'] ** begin_epoch
        params = list()
        if up_sampler or down_sampler:
            if up_sampler is not None:
                sr_optimizer = torch.optim.Adam(sr_model.parameters(), lr=lr)
                sr_scheduler = torch.optim.lr_scheduler.ExponentialLR(sr_optimizer,
                                                                      gamma=train_params['decay_rate'],
                                                                      last_epoch=begin_epoch - 1)
            if down_sampler is not None:
                ds_optimizer = torch.optim.Adam(ds_model.parameters(), lr=lr)
                ds_scheduler = torch.optim.lr_scheduler.ExponentialLR(ds_optimizer,
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
            batch_size=train_params['batch_size'] if is_train else 1,
            shuffle=True,
            num_workers=train_params['num_worker']
        )
    val_dataset = SRTestDataset(hr_dir=val_hr_dir, lr_dir=val_lr_dir)
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_params['batch_size'] if is_train else 1,
        shuffle=True if is_train else False,
        num_workers=train_params['num_worker']
    )

    # Training loop and saver as checkpoints
    print('Using device {}'.format(device))
    print(sr_model.splitter)
    timer = Timer()
    cnt = 0
    print(sr_model.t)
    print(sr_model.splitter)
    best_val = None
    for epoch in range(begin_epoch, num_epoch):
        sr_model.train_step(train_loader, sr_optimizer, sr_scheduler, sr_loss)
        torch.cuda.empty_cache()
        val_l = sr_model.test_step(val_dataset)
        if best_val is None or best_val > val_l:
            sr_model.save_checkpoint()

    '''
    with open(log_dir, 'a') as f:
        for epoch in range(begin_epoch, num_epoch):
            epoch_ls, epoch_sr, epoch_lr, epoch_diff = [], [], [], []
            sr_l, ds_l, sr_psnr, ds_psnr = -1., -1., -1., -1.
            for bid, batch in enumerate(train_loader):
                hr, lr, ls = batch['hr'].cuda(), batch['lr'].cuda(), list()
                timer.refresh()

                if is_train:
                    if up_sampler is not None:
                        sr_optimizer.zero_grad()
                    if down_sampler is not None:
                        ds_optimizer.zero_grad()
                else:
                    torch.cuda.empty_cache()
                    
                if up_sampler is not None:
                    sr = sr_model(lr)
                    if is_train:
                        sr_l = sr_loss(sr, hr)
                        ls.append(sr_l)
                    else:
                        sr_ot = np.moveaxis(sr.detach().cpu().numpy().squeeze(0), 0, -1).astype(np.uint8)
                        imwrite(os.path.join(sr_dir, batch['name'][0]), sr_ot)
                    sr_psnr = psnr(sr, hr, trim=trim).detach().cpu().item()
                    epoch_sr.append(sr_psnr)

                if down_sampler is not None:
                    dhr = ds_model(hr)
                    if is_train:
                        ds_l = ds_loss(dhr, lr)
                        ls.append(pipeline_params['ds_beta'] * ds_l)
                    else:
                        pass
                    ds_psnr = psnr(dhr, lr).detach().cpu().item()
                    epoch_lr.append(ds_psnr)
                else:
                    dsl = ds_loss(sr, lr)
                    if is_train:
                        ls.append(pipeline_params['ds_beta'] * dsl)

                l = sum(ls)
                epoch_ls.append(l)
                if is_train:
                    l.backward()
                    if up_sampler is not None:
                        sr_optimizer.step()
                    if down_sampler is not None:
                        ds_optimizer.step()

                ep_l = sum(epoch_ls) / (bid + 1)
                ep_sr = sum(epoch_sr) / (bid + 1)
                ep_lr = sum(epoch_lr) / (bid + 1)
                ep_df = sum(epoch_diff) / (bid + 1)
                duration = timer.report()

                report = report_formatter.format(epoch, bid, l, ep_l, sr_psnr, ep_sr, dsl, ep_lr, ep_df, duration)
                if bid % pipeline_params['print_every'] == 0:
                    print(report)
                    print(title, end='\r')

            if is_train:
                f.write(report + '\n')
                f.flush()
                if up_sampler is not None:
                    sr_scheduler.step()
                if down_sampler is not None:
                    ds_scheduler.step()
                if epoch % pipeline_params['save_every'] == 0 or epoch == pipeline_params['num_epoch'] - 1:
                    if up_sampler is not None:
                        sr_model.save_checkpoint()
                    if down_sampler is not None:
                        ds_model.save_checkpoint()
            else:
                pass
            print(splitter)
    '''
