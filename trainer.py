import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from helper import *
from dataset import SRTrainDataset
import os, sys, math
import getopt
from time import time

if __name__ == '__main__':
    args = sys.argv[1:]
    long_opts = [
        'model-name=', 'hr-dir=', 'lr-dir=', 'output-dir=', 'upsample=', 'learning-rate=', 'decay-rate=', 'decay-step=',
        'batch-size=', 'num-epoch=', 'add-dsloss=', 'gpu-device=', 'log-dir=', 'ckpt-dir=', 'window='
    ]

    try:
        optlist, args = getopt.getopt(args, '', long_opts)
    except getopt.GetoptError as e:
        print(str(e))
        sys.exit(2)
    for arg, opt in optlist:
        if arg == '--model-name':
            model_name = str(opt)
            model = load_model(model_name)
        elif arg == '--hr-dir':
            HR_DIR = str(opt)
            if not os.path.exists(HR_DIR):
                raise FileNotFoundError('High resolution images not found.')
        elif arg == '--lr-dir':
            LR_DIR = str(opt)
            if not os.path.exists(LR_DIR):
                raise FileNotFoundError('Low resolution images not found.')
        elif arg == '--output-dir':
            OUT_DIR = str(opt)
            if not os.path.exists(OUT_DIR):
                os.mkdir(OUT_DIR)
            OUT_DIR = os.path.join(OUT_DIR, model_name)
            if not os.path.exists(OUT_DIR):
                os.mkdir(OUT_DIR)
        elif arg == '--ckpt-dir':
            CKPT_DIR = str(opt)
            if not os.path.exists(CKPT_DIR):
                os.mkdir(CKPT_DIR)
            CKPT_DIR = os.path.join(CKPT_DIR, model_name)
            if not os.path.exists(CKPT_DIR):
                os.mkdir(CKPT_DIR)
        elif arg == '--log-dir':
            LOG_DIR = str(opt)
            if not os.path.exists(LOG_DIR):
                os.mkdir(LOG_DIR)

        elif arg == '--learning-rate':
            LEARNING_RATE = float(opt)
        elif arg == '--decay-rate':
            DECAY_RATE = float(opt)
        elif arg == '--decay-step':
            DECAY_STEP = int(opt)
        elif arg == '--batch-size':
            BATCH_SIZE = int(opt)
        elif arg == '--num-epoch':
            NUM_EPOCH = int(opt)

        elif arg == '--gpu-device':
            if torch.cuda.is_available:
                DEVICES = [int(i) for i in ','.split(opt)]
            DEVICE = torch.device(str(DEVICES[0]) if torch.cuda.is_available else 'cpu')

        elif arg == '--window':
            WIN_SIZE = int(opt)
        else:
            continue

    # initialize the SRResNet model and loss function and optimizer and learning rate scheduler
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    # model = model.to(DEVICE)
    loss = MSEnDSLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_STEP)
    dataset = SRTrainDataset(HR_DIR, LR_DIR, h=WIN_SIZE, w=WIN_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    begin_epoch = 0

    # load checkpoint
    ckpt = load_checkpoint(load_dir=CKPT_DIR, map_location=None, model_name=model_name)
    if ckpt is not None:
        print('recovering from checkpoints...')
        model.load_state_dict(ckpt['model'])
        begin_epoch = ckpt['epoch'] + 1
        print('resuming training')

    # training loop
    begin = time()
    with open(os.path.join(LR_DIR, '{}.log'.format(model_name)), 'a') as f:
        for epoch in range(begin_epoch, NUM_EPOCH):
            epoch_loss = []
            for bid, batch in enumerate(dataloader):
                hr, lr = batch['hr'].to(DEVICE), batch['lr'].to(DEVICE)
                optimizer.zero_grad()
                x = model(lr)
                batch_loss = loss(x, hr, lr)
                batch_loss.backward()
                optimizer.step()
                epoch_loss.append(batch_loss)

                print('Epoch {} | Batch {} | Bpsnr {:6f} | Epsnr {:6f} | RT {:6f}'.format(epoch, bid, psnr(batch_loss),
                                                                                          psnr(np.mean(epoch_loss)),
                                                                                          since(begin)))
                f.write('{},{},{},{},{}\n'.format(epoch, bid, batch_loss, np.mean(epoch_loss), since(begin)))
                f.flush()
            state_dict = {
                'model': model.state_dict(),
                'epoch': epoch
            }
            save_checkpoint(state_dict, CKPT_DIR, model_name=model_name)
