import torch
import torch.nn as nn
from src.dataset import ImageDataset
from torch.utils.data import DataLoader
from model.downscaler.conv import ConvolutionDownscale
from src.helper import load_checkpoint, save_checkpoint, since
import numpy as np
from time import time
import os
DEVICE = torch.device('cuda:2' if torch.cuda.is_available else 'cpu')

if __name__ == '__main__':
    model = ConvolutionDownscale()
    print(DEVICE, torch.cuda.is_available)
    model = model.to(DEVICE)
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=.96)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000)
    dataset = ImageDataset(hr_dir='/usr/project/xtmp/superresoluter/dataset/DIV2K/DIV2K_train_HR/',
                           lr_dir='/usr/project/xtmp/superresoluter/dataset/DIV2K/DIV2K_train_LR_bicubic/X4',
                           lr_parse=lambda x: x.replace('x4', ''))
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    begin_epoch = 0

    ckpt = load_checkpoint(load_dir='./checkpoints/', map_location=None, model_name='down_sample')
    if ckpt is not None:
        print('recovering from checkpoints...')
        model.load_state_dict(ckpt['model'])
        begin_epoch = ckpt['epoch'] + 1
        print('resuming training')

    begin = time()
    with open(os.path.join('../logs', 'down_sample.log'), 'w') as f:
        for epoch in range(begin_epoch, 1000):
            epoch_loss = []
            for bid, batch in enumerate(loader):
                hr, lr = batch['hr'].to(DEVICE), batch['lr'].to(DEVICE)
                optimizer.zero_grad()
                ds = model(hr)
                batch_loss = loss(ds, lr)
                batch_loss.backward()
                optimizer.step()
                epoch_loss.append(batch_loss.cpu().detach().numpy())
                print('Epoch {} | Batch {} | BMSE {:6f} | EMSE {:.6f} | RT {:6f}'.format(epoch, bid, batch_loss,
                                                                                         np.mean(epoch_loss),
                                                                                         since(begin)))
                f.write('{},{},{},{},{}\n'.format(epoch, bid, batch_loss, np.mean(epoch_loss), since(begin)))
                f.flush()
            state_dict = {
                'model': model.state_dict(),
                'epoch': epoch
            }
            save_checkpoint(state_dict, '../checkpoints/', model_name='down_sample')
