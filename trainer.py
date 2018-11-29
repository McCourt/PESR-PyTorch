import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from helper import *
from dataset import ResolutionDataset
import os, sys, math
import getopt
from time import time


if __name__ == '__main__':
    args = sys.argv[1:]
    longopts = [
        'model-name=', 'hr-dir=', 'lr-dir=', 'output-dir=', 'upsample=',
        'learning-rate=', 'decay-rate=', 'decay-step=', 'batch-size=',
        'num-epoch=', 'add-dsloss=', 'gpu-device=', 'log-dir=', 'ckpt-dir=',
        'window='
    ]

    try:
        optlist, args = getopt.getopt(args, '', longopts)
    except getopt.GetoptError as e:
        print(str(e))
        sys.exit(2)
    for arg, opt in optlist:
        if arg == '--model-name':
            model_name = str(opt)
            model = load_model(model_name)
        elif arg == '--hr-dir':
            HR_DIR = str(opt)
        elif arg == '--lr-dir':
            LR_DIR = str(opt)
        elif arg == '--output-dir':
            OUT_DIR = str(opt)
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
            DEVICE = torch.device(str(opt) if torch.cuda.is_available else 'cpu')
        elif arg == '--log-dir':
            LOG_PATH = str(opt)
        elif arg == '--add-dsloss':
            ADD_DS = False if str(opt).lower() == 'false' else True
        elif arg == '--ckpt-dir':
            CKPT_DIR = str(opt)
        elif arg == '--upsample':
            UPSAMPLE = False if str(opt).lower() == 'false' else True
        elif arg == '--window':
            WIN_SIZE = int(opt)
        else:
            continue

    ## initialize the SRResNet model and loss function and optimizer and learning rate scheduler
    model = model.to(DEVICE)
    loss = MSEnDSLoss(add_ds=ADD_DS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_STEP)
    dataset = ResolutionDataset(HR_DIR, LR_DIR, upsample=UPSAMPLE, h=WIN_SIZE, w=WIN_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    begin_epoch = 0

    ## load checkpoint
    ckpt = load_checkpoint(load_dir=CKPT_DIR, map_location=None, model_name=model_name)
    if ckpt is not None:
        print('recovering from checkpoints...')
        model.load_state_dict(ckpt['model'])
        begin_epoch = ckpt['epoch'] + 1
        print('resuming training')

    ## training loop
    begin = time()
    with open(os.path.join(LOG_PATH, '{}.log'.format(model_name)), 'a') as f:
        for epoch in range(begin_epoch, NUM_EPOCH):
            epoch_loss = 0
            for bid, batch in enumerate(dataloader):
                hr, lr = batch['hr'].to(DEVICE), batch['lr'].to(DEVICE)
                optimizer.zero_grad()
                x = model(lr)
                l = loss(x, hr, lr)
                l.backward()
                optimizer.step()
                epoch_loss += l

                print('Epoch {} | Batch {} | Bpsnr {:6f} | Epsnr {:6f} | RT {:6f}'.format(epoch, bid, psnr(l), psnr(epoch_loss / (1 + bid)), since(begin)))
                f.write('{},{},{},{},{}\n'.format(epoch, bid, l, epoch_loss / (1 + bid), since(begin)))
                f.flush()
            state_dict = {
                'model': model.state_dict(),
                'epoch': epoch
            }
            save_checkpoint(state_dict, CKPT_DIR, model_name=model_name)
