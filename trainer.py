from torch.utils.data import DataLoader
from src.helper import *
from src.dataset import SRTrainDataset, SRTestDataset
from loss.psnr import PSNR
from loss.dsloss import DownScaleLoss
import os, sys
import getopt
from time import time

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
        params = load_parameters()
        print('Parameters loaded')
        print(''.join(['-' for i in range(30)]))
        pipeline_params, common_params = params[mode], params['common']
        for i in sorted(pipeline_params):
            print('{:<15s} -> {}'.format(str(i), pipeline_params[i]))
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
    model_name = '+'.join([str(i) for i in [up_sampler, down_sampler]])
    trim, scale, begin_epoch, psnr = common_params['trim'], common_params['scale'], 0, PSNR()
    hr_dir = os.path.join(root_dir, common_params['s0_dir'], pipeline_params['hr_dir'])
    lr_dir = os.path.join(root_dir, common_params['s0_dir'], pipeline_params['lr_dir'])
    sr_dir = os.path.join(root_dir, common_params['s1_dir'], pipeline_params['sr_dir'])
    log_dir = os.path.join(root_dir, common_params['log_dir'].format(model_name))
    num_epoch = pipeline_params['num_epoch'] if mode == 'train' else 1 + begin_epoch

    # Define upscale model and data parallel
    if up_sampler is not None:
        sr_model = load_model(up_sampler)
        sr_model = nn.DataParallel(sr_model).cuda()

        try:
            sr_ckpt = os.path.join(root_dir, common_params['ckpt_dir'].format(up_sampler))
            sr_checkpoint = load_checkpoint(load_dir=sr_ckpt, map_location=pipeline_params['map_location'])
            if sr_checkpoint is None:
                print('Start new training for SR model')
            else:
                print('SR model recovering from checkpoints')
                sr_model.load_state_dict(sr_checkpoint['model'])
        except Exception as e:
                raise ValueError('Checkpoint not found.')

        print('Number of parameters of SR model: {:.2E}'.format(sum(p.numel() for p in sr_model.parameters() if p.requires_grad)))
        for param in sr_model.parameters():
            param.requires_grad = False if mode == 'test' else param.requires_grad
        sr_loss = nn.L1Loss().cuda()

    # Define downscale model and data parallel and loss functions
    if down_sampler is not None:
        ds_model = load_model(down_sampler)
        ds_model = nn.DataParallel(ds_model).cuda()

        try:
            ds_ckpt = os.path.join(root_dir, common_params['ckpt_dir'].format(down_sampler))
            ds_checkpoint = load_checkpoint(load_dir=ds_ckpt, map_location=pipeline_params['map_location'])
            if down_sampler is None or ds_checkpoint is None:
                print('Start new training for DS model')
            else:
                print('DS model recovering from checkpoints')
                ds_model.load_state_dict(ds_checkpoint['model'])
        except Exception as e:
                raise ValueError('Checkpoint not found.')

        print('Number of parameters of DS model: {:.2E}'.format(sum(p.numel() for p in ds_model.parameters())))
        for param in ds_model.parameters():
            param.requires_grad = False if mode == 'test' else param.requires_grad
        ds_loss = nn.MSELoss().cuda()
    else:
        bds = DownScaleLoss(clip_round=True)

    # Define optimizer, learning rate scheduler, data source and data loader
    if mode == 'train':
        params = list()
        if up_sampler or down_sampler:
            if up_sampler is not None:
                sr_optimizer = torch.optim.Adam(sr_model.parameters(), lr=pipeline_params['learning_rate'])
                sr_scheduler = torch.optim.lr_scheduler.ExponentialLR(sr_optimizer,
                                                                      gamma=pipeline_params['decay_rate'],
                                                                      last_epoch=begin_epoch - 1)
            if down_sampler is not None:
                ds_optimizer = torch.optim.Adam(ds_model.parameters(), lr=pipeline_params['learning_rate'])
                ds_scheduler = torch.optim.lr_scheduler.ExponentialLR(ds_optimizer,
                                                                      gamma=pipeline_params['decay_rate'],
                                                                      last_epoch=begin_epoch - 1)
        else:
            raise Exception('No trainable parameters in training mode')

        dataset = SRTrainDataset(
            hr_dir=hr_dir,
            lr_dir=lr_dir,
            h=pipeline_params['window'][0],
            w=pipeline_params['window'][1],
            scale=scale,
            num_per=pipeline_params['num_per']
        )
    else:
        dataset = SRTestDataset(hr_dir=hr_dir, lr_dir=lr_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=pipeline_params['batch_size'] if mode == 'train' else 1,
        shuffle=True if mode == 'train' else False,
        num_workers=pipeline_params['num_worker']
    )

    # Training loop and saver as checkpoints
    print('Using device {}'.format(device))
    title_formatter = '{:^6s} | {:^6s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^10s} '
    report_formatter = '{:^6d} | {:^6d} | {:^8.4f} | {:^8.4f} | {:^8.4f} | {:^8.4f} | {:^8.4f} | {:^8.4f} | {:^8.4f} | {:^10.4f} '
    title = title_formatter.format('Epoch', 'Batch', 'BLoss', 'ELoss', 'SR_PSNR', 'AVG_SR',
                                   'DS_PSNR', 'AVG_DS', 'AVG_DIFF', 'RunTime')
    splitter = ''.join(['-' for i in range(len(title))])
    print(splitter)
    begin = time()
    cnt = 0
    print(title)
    print(splitter)
    with open(log_dir, 'a') as f:
        for epoch in range(begin_epoch, num_epoch):
            epoch_ls, epoch_sr, epoch_lr, epoch_diff = [], [], [], []
            sr_l, ds_l, sr_psnr, ds_psnr = -1., -1., -1., -1.
            for bid, batch in enumerate(data_loader):
                hr, lr = batch['hr'].cuda(), batch['lr'].cuda()
                ls = list()
                if mode == 'train':
                    if up_sampler is not None:
                        sr_optimizer.zero_grad()
                    if down_sampler is not None:
                        ds_optimizer.zero_grad()
                else:
                    torch.cuda.empty_cache()
                    
                if up_sampler is not None:
                    sr = sr_model(lr)
                    if mode == 'train':
                        sr_l = sr_loss(sr, hr)
                        ls.append(sr_l)
                    sr_psnr = psnr(sr, hr, trim=trim).detach().cpu().item()
                    epoch_sr.append(sr_psnr)

                if down_sampler is not None:
                    dhr = ds_model(hr)
                    if mode == 'train':
                        ds_l = ds_loss(dhr, lr)
                        ls.append(pipeline_params['ds_beta'] * ds_l)
                    ds_psnr = psnr(dhr, lr).detach().cpu().item()
                    epoch_lr.append(ds_psnr)
                else:
                    if mode == 'train':
                        dsl = bds(sr, lr).detach().cpu().item()
                        ls.append(pipeline_params['ds_beta'] * dsl)
                    else:
                        dsl = 0.0

                l = sum(ls)
                epoch_ls.append(l)
                if mode == 'train':
                    l.backward()
                    if up_sampler is not None:
                        sr_optimizer.step()
                    if down_sampler is not None:
                        ds_optimizer.step()

                ep_l = sum(epoch_ls) / (bid + 1)
                ep_sr = sum(epoch_sr) / (bid + 1)
                ep_lr = sum(epoch_lr) / (bid + 1)
                ep_df = sum(epoch_diff) / (bid + 1)
                timer = since(begin)

                report = report_formatter.format(epoch, bid, l, ep_l, sr_psnr, ep_sr, dsl, ep_lr, ep_df, timer)
                if bid % pipeline_params['print_every'] == 0:
                    print(report)
                    print(title, end='\r')

                if mode == 'test':
                    del hr
                    del lr
                    del sr
                    torch.cuda.empty_cache()

            if mode == 'train':
                f.write(report + '\n')
                f.flush()
                if up_sampler is not None:
                    sr_scheduler.step()
                if down_sampler is not None:
                    ds_scheduler.step()
                if epoch % pipeline_params['save_every'] == 0 or epoch == pipeline_params['num_epoch'] - 1:
                    if up_sampler is not None:
                        state_dict = {'model': sr_model.state_dict()}
                        save_checkpoint(state_dict, sr_ckpt)
                    if down_sampler is not None:
                        state_dict = {'model': ds_model.state_dict()}
                        save_checkpoint(state_dict, ds_ckpt)
            elif mode == 'test':
                pass
            print(splitter)
