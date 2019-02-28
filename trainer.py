from torch.utils.data import DataLoader
from src.helper import *
from src.dataset import SRTrainDataset
import os, sys
import getopt
from time import time

if __name__ == '__main__':
    print('{} GPUS Available'.format(torch.cuda.device_count()))

    # Load system arguments
    args = sys.argv[1:]
    long_opts = ['model-name=', 'scale=']

    try:
        optlist, args = getopt.getopt(args, '', long_opts)
    except getopt.GetoptError as e:
        print(str(e))
        sys.exit(2)

    for arg, opt in optlist:
        if arg == '--model-name':
            model_name = str(opt)
            sr_model = load_model(model_name)
            # ds_model = load_model(model_name)
        elif arg == '--scale':
            scale = int(opt)
        else:
            raise Exception("Redundant Argument Detected")

    # Load JSON arguments
    try:
        params = load_parameters()
        print('Parameters loaded')
        print(''.join(['-' for i in range(30)]))
        train_params = params['train']
        common_params = params['common']
        for i in train_params:
            print('{:<15s} -> {}'.format(str(i), train_params[i]))
    except Exception as e:
        print(e)
        raise ValueError('Parameter not found.')

    # Prepare all directory and devices
    device = torch.device('cuda:{}'.format(train_params['device_ids']) if torch.cuda.is_available else 'cpu')
    lr_dir = os.path.join(common_params['root_dir'], common_params['lr_dir'])
    hr_dir = os.path.join(common_params['root_dir'], common_params['hr_dir'])
    sr_dir = os.path.join(common_params['root_dir'], common_params['sr_dir'].format(model_name))
    log_dir = os.path.join(common_params['root_dir'], common_params['train_log'].format(model_name))
    ckpt_dir = os.path.join(common_params['root_dir'], common_params['ckpt_dir'].format(model_name))

    # Define model and loss function
    sr_model = nn.DataParallel(sr_model).cuda()
    sr_loss = nn.MSELoss(reduction='elementwise_mean').to(device)
    lr_loss = nn.MSELoss(reduction='elementwise_mean').to(device)
    lr_loss.require_grad = False
    sr_model = sr_model.cuda()

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(
        sr_model.parameters(),
        lr=train_params['learning_rate']
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=train_params['decay_rate']
    )

    # Define data source
    dataset = SRTrainDataset(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        h=train_params['window'][0],
        w=train_params['window'][1],
        scale=scale,
        num_per=train_params['num_per']
    )
    data_loader = DataLoader(
        dataset,
        batch_size=train_params['batch_size'],
        shuffle=True,
        num_workers=train_params['num_worker']
    )

    # Load checkpoint
    begin_epoch = 0
    ckpt = load_checkpoint(
        load_dir=ckpt_dir,
        map_location=train_params['map_location']
    )
    try:
        if ckpt is None:
            print('Start new training')
        else:
            print('recovering from checkpoints', end='\r')
            sr_model.load_state_dict(ckpt['model'])
            begin_epoch = ckpt['epoch'] + 1
            print('resuming training from epoch {}'.format(begin_epoch))
    except Exception as e:
        raise ValueError('Checkpoint not found.')

    # Training loop and saver as checkpoints
    print('Using device {}'.format(device))
    title_formatter = '{:^6s} | {:^6s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^10s} '
    report_formatter = '{:^6d} | {:^6d} | {:^8.2f} | {:^8.2f} | {:^8.2f} | {:^8.2f} | {:^8.2f} | {:^10.2f}'
    title = title_formatter.format('Epoch', 'Batch', 'BLoss', 'ELoss', 'BPSNR', 'EPSNR', 'RPSNR', 'Runtime')
    splitter = ''.join(['-' for i in range(len(title))])
    print(splitter)
    begin = time()
    cnt = 0
    print(title)
    print(splitter)
    with open(log_dir, 'w') as f:
        for epoch in range(begin_epoch, train_params['num_epoch']):
            e_ls = []
            for bid, batch in enumerate(data_loader):
                hr, lr = batch['hr'].to(device), batch['lr'].to(device)
                optimizer.zero_grad()
                sr = sr_model(lr)
                b_loss = sr_loss(sr, hr)
                r_loss = lr_loss(lr, hr)
                b_loss.backward()
                optimizer.step()
                e_ls.append(b_loss)
                e_loss = sum(e_ls) / (bid + 1)
                report = report_formatter.format(epoch, bid, b_loss, e_loss, psnr(b_loss),
                                                 psnr(e_loss), psnr(r_loss), since(begin))
                if bid % train_params['print_every'] == 0:
                    print(report)
                    print(title, end='\r')

                f.write(report + '\n')
                f.flush()
            scheduler.step()
            if epoch % train_params['save_every'] == 0 or epoch == train_params['num_epoch'] - 1:
                state_dict = {
                    'model': sr_model.state_dict(),
                    'epoch': epoch
                }
                save_checkpoint(state_dict, ckpt_dir)
            print(splitter)
