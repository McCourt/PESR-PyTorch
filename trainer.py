from torch.utils.data import DataLoader
from src.helper import *
from src.dataset import SRTrainDataset, SRTestDataset
import os, sys
import getopt
from time import time

if __name__ == '__main__':
    print('{} GPUS Available'.format(torch.cuda.device_count()))

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
        pipeline_params = params[mode]
        common_params = params['common']
        for i in pipeline_params:
            print('{:<15s} -> {}'.format(str(i), pipeline_params[i]))
    except Exception as e:
        print(e)
        raise ValueError('Parameter not found.')

    # Prepare common parameters
    device = torch.device('cuda:{}'.format(common_params['device_ids']) if torch.cuda.is_available else 'cpu')
    root_dir = common_params['root_dir']
    if common_params['up_sampler'] is not None:
        up_sampler = common_params['up_sampler']
    else:
        raise Exception('You must define a upscale model for super resolution')
    down_sampler = common_params['down_sampler']
    scale = common_params['scale']

    # Prepare all directory and devices
    lr_dir = os.path.join(root_dir, common_params['lr_dir'])
    hr_dir = os.path.join(root_dir, common_params['hr_dir'])
    sr_dir = os.path.join(root_dir, common_params['sr_dir'].format(up_sampler))
    log_dir = os.path.join(root_dir, common_params['log_dir'].format(mode, up_sampler))
    sr_ckpt = os.path.join(root_dir, common_params['ckpt_dir'].format(up_sampler))
    if down_sampler is not None:
        ds_ckpt = os.path.join(root_dir, common_params['ckpt_dir'].format(down_sampler))

    # Define upscale model and data parallel
    sr_model = load_model(up_sampler)
    sr_model = nn.DataParallel(sr_model).cuda()

    # Define downscale model and data parallel
    if down_sampler is not None:
        ds_model = load_model(down_sampler)
        ds_model = nn.DataParallel(ds_model).cuda()

    # Define loss functions
    sr_loss = nn.MSELoss(reduction='elementwise_mean').to(device)
    if down_sampler is not None:
        ds_loss = nn.MSELoss(reduction='elementwise_mean').to(device)
    real_loss = nn.MSELoss(reduction='elementwise_mean').to(device)
    real_loss.require_grad = False

    # Define optimizer and learning rate scheduler
    params = sr_model.parameters()
    if down_sampler is not None:
        params = list(params) + list(ds_model.parameters())
    optimizer = torch.optim.Adam(
        params,
        lr=pipeline_params['learning_rate']
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=pipeline_params['decay_rate']
    )

    # Define data source and data loader
    if mode == 'train':
        dataset = SRTrainDataset(
            hr_dir=hr_dir,
            lr_dir=lr_dir,
            h=pipeline_params['window'][0],
            w=pipeline_params['window'][1],
            scale=scale,
            num_per=pipeline_params['num_per']
        )
    else:
        dataset = SRTestDataset(
            hr_dir=hr_dir,
            lr_dir=lr_dir
        )
    data_loader = DataLoader(
        dataset,
        batch_size=pipeline_params['batch_size'] if mode == 'train' else 1,
        shuffle=True if mode == 'train' else False,
        num_workers=pipeline_params['num_worker']
    )

    # Load checkpoint
    begin_epoch = 0
    sr_checkpoint = load_checkpoint(
        load_dir=sr_ckpt,
        map_location=pipeline_params['map_location']
    )
    if down_sampler is not None:
        ds_checkpoint = load_checkpoint(
            load_dir=ds_ckpt,
            map_location=pipeline_params['map_location']
        )
    try:
        if sr_checkpoint is None:
            print('Start new training')
        else:
            print('recovering from checkpoints', end='\r')
            sr_model.load_state_dict(sr_checkpoint['model'])
            if down_sampler is not None:
                ds_model.load_state_dict(ds_checkpoint['model'])
            begin_epoch = sr_checkpoint['epoch'] + 1
            print('resuming training from epoch {}'.format(begin_epoch))
    except Exception as e:
        raise ValueError('Checkpoint not found.')

    # Training loop and saver as checkpoints
    print('Using device {}'.format(device))
    title_formatter = '{:^6s} | {:^6s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^10s} '
    report_formatter = '{:^6d} | {:^6d} | {:^8.2f} | {:^8.2f} | {:^8.2f} | {:^8.2f} | {:^8.2f} | {:^8.2f} | {:^8.2f} ' \
                       '| {:^10.2f} '
    title = title_formatter.format('Epoch', 'Batch', 'BLoss', 'ELoss', 'SR_PSNR', 'AVG_PSNR',
                                   'DS_PSNR', 'AVG_PSNR', 'R_PSNR', 'RunTime')
    splitter = ''.join(['-' for i in range(len(title))])
    print(splitter)
    begin = time()
    cnt = 0
    print(title)
    print(splitter)
    with open(log_dir, 'w') as f:
        for epoch in range(begin_epoch, pipeline_params['num_epoch']):
            epoch_ls, epoch_sr, epoch_lr = [], [], []
            ds_l, ds_psnr = -1., -1.
            for bid, batch in enumerate(data_loader):
                hr, lr = batch['hr'].to(device), batch['lr'].to(device)
                optimizer.zero_grad()

                sr = sr_model(lr)
                sr_l = sr_loss(sr, hr)
                sr_psnr = psnr(sr_l)
                epoch_sr.append(sr_psnr)

                if down_sampler is not None:
                    dsr = ds_model(sr)
                    ds_l = ds_loss(dsr, lr)
                    l = ds_l + sr_l
                    ds_psnr = psnr(ds_l)
                else:
                    l = sr_l
                epoch_lr.append(ds_psnr)

                real_l = real_loss(lr, hr)
                real_psnr = psnr(real_l)

                l.backward()
                optimizer.step()
                epoch_ls.append(l)

                ep_l = sum(epoch_ls) / (bid + 1)
                ep_sr = sum(epoch_sr) / (bid + 1)
                ep_lr = sum(epoch_lr) / (bid + 1)

                timer = since(begin)

                report = report_formatter.format(epoch, bid, l, ep_l, sr_psnr, ep_sr, ds_psnr, ep_lr, real_psnr, timer)
                if bid % pipeline_params['print_every'] == 0:
                    print(report)
                    print(title, end='\r')

                f.write(report + '\n')
                f.flush()
            scheduler.step()

            if epoch % pipeline_params['save_every'] == 0 or epoch == pipeline_params['num_epoch'] - 1:
                state_dict = {
                    'model': sr_model.state_dict(),
                    'epoch': epoch
                }
                save_checkpoint(state_dict, sr_ckpt)
                if down_sampler is not None:
                    state_dict = {
                        'model': ds_model.state_dict(),
                    }
                    save_checkpoint(state_dict, ds_ckpt)
            print(splitter)
