import torch
import torch.nn as nn
from dataset import ImageDataset
from torch.utils.data import DataLoader
from downsample.conv import ConvolutionDownscale
from helper import load_checkpoint, save_checkpoint, psnr, since
import numpy as np
from time import time
import os


if __name__ == '__main__':
    model = ConvolutionDownscale()
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=.96)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000)
    dataset = ImageDataset(HR_DI='/usr/project/xtmp/superresoluter/dataset/DIV2K/DIV2K_train_HR/',
                           LR_DIR='/usr/project/xtmp/superresoluter/dataset/DIV2K/DIV2K_train_LR_bicubic//')
    loader = DataLoader(dataset, batch_size=40, shuffle=True, num_workers=8)
    begin_epoch = 0

    ckpt = load_checkpoint(load_dir='', map_location=None, model_name='down_sample')
    if ckpt is not None:
        print('recovering from checkpoints...')
        model.load_state_dict(ckpt['model'])
        begin_epoch = ckpt['epoch'] + 1
        print('resuming training')

    begin = time()
    with open(os.path.join('./', 'down_sample.log'), 'w') as f:
        for epoch in range(begin_epoch, 30):
            epoch_loss = []
            for bid, batch in enumerate(loader):
                hr, lr = batch['hr'], batch['lr']
                optimizer.zero_grad()
                ds = model(hr)
                batch_loss = loss(ds, lr)
                batch_loss.backward()
                optimizer.step()
                epoch_loss.append(batch_loss)

                print('Epoch {} | Batch {} | Bpsnr {:6f} | Epsnr {:6f} | RT {:6f}'.format(epoch, bid, batch_loss,
                                                                                          np.mean(epoch_loss),
                                                                                          since(begin)))
                f.write('{},{},{},{},{}\n'.format(epoch, bid, batch_loss, np.mean(epoch_loss), since(begin)))
                f.flush()
            state_dict = {
                'model': model.state_dict(),
                'epoch': epoch
            }
            save_checkpoint(state_dict, './checkpoints/', model_name='down_sample')