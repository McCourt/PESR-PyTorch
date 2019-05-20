import torch
import torch.nn as nn
from importlib import import_module
from src.helper import load_parameters, Timer
from loss.psnr import PSNR
import os
import numpy as np


def save_checkpoint(state_dict, save_dir):
    try:
        torch.save(state_dict, save_dir)
    except:
        raise Exception('checkpoint saving failure')


def load_checkpoint(load_dir, map_location=None):
    try:
        print('loading checkpoint from {}'.format(load_dir))
        checkpoint = torch.load(load_dir, map_location=map_location)
        print('loading successful')
        return checkpoint
    except:
        print('No checkpoint and begin new training')


def report_num_params(model):
    print('Number of parameters of model: {:.2E}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))


class Model(nn.Module):
    def __init__(self, name, mode, checkpoint=None, train=True, map_location=None, log=None, **kwargs):
        super().__init__()
        params = load_parameters(path='model/models.json')

        if mode not in params.keys():
            raise ValueError('Wrong mode. Try {}'.format(', '.join(params.keys())))
        elif name not in params[mode]:
            raise ValueError('Wrong model name. Try {}'.format(', '.join(params[mode].keys())))
        path = '.'.join(['model', mode])
        module = getattr(import_module(path), params[mode][name.lower()])
        self.model = module(**kwargs)
        self.model = nn.DataParallel(self.model).cuda()
        report_num_params(self.model)

        self.checkpoint, self.mode, self.map_location, self.log = checkpoint, mode, map_location, log
        self.epoch = 0
        self.load_checkpoint()
        self.timer = Timer()

        self.is_train = train
        if not self.is_train:
            print('Disabling auto gradient and switching to TEST mode')
            self.train()
        else:
            self.eval()
            print('{} model is ready for training'.format(mode))
        self.metric = PSNR()
        self.t_format = '{:^6s} | {:^6s} | {:^7s} | {:^7s} | {:^7s} | {:^7s} | {:^8s} '
        self.r_format = '{:^6d} | {:^6d} | {:^.4f} | {:^.4f} | {:^.4f} | {:^.4f} | {:^.4E} '
        self.t = self.t_format.format('Epoch', 'Batch', 'BLoss', 'ELoss', 'PSNR', 'AVGPSNR', 'Runtime')
        self.splitter = ''.join(['-' for i in range(len(self.t))])

    def load_checkpoint(self):
        if self.checkpoint is not None and os.path.isfile(self.checkpoint):
            try:
                print('loading checkpoint from {}'.format(self.checkpoint))
                ckpt = torch.load(self.checkpoint, map_location=self.map_location)
                if ckpt is None:
                    print('No checkpoint and start new training for {} model'.format(self.mode))
                else:
                    print('loading successful and recovering checkpoints for {} model'.format(self.mode))
                    self.load_state_dict(ckpt)
                    print('Checkpoint loaded successfully')
            except:
                print('Checkpoint failed to load, continuing without pretrained checkpoint')
                #raise ValueError('Wrong Checkpoint path or loaded erroneously')
        else:
            print('No checkpoint and start new training for {} model'.format(self.mode))

    def save_checkpoint(self, add_time=False):
        try:
            torch.save(self.state_dict(), self.checkpoint)
        except:
            raise Exception('checkpoint saving failed')

    def forward(self, x, clip_round=False):
        output = self.model(x)
        if clip_round:
            output = torch.clamp(torch.round(output), 0., 255.)
        return output

    def train_step(self, data_loader, optimizer, scheduler, loss_fn):
        self.train()
        ls, ps = list(), list()
        for bid, batch in enumerate(data_loader):
            hr, lr = batch['hr'].cuda(), batch['lr'].cuda()
            optimizer.zero_grad()
            sr = self.forward(lr)

            l = loss_fn(sr, hr)
            ls.append(l)
            psnr = self.metric(sr, hr).detach().cpu().item()
            ps.append(psnr)
            print(self.r_format.format(self.epoch, bid, l, sum(ls) / len(ls),
                                       psnr, sum(ps) / len(ps), self.timer.report()))
            print(self.t, end='\r')

            l.backward()
            optimizer.step()
            self.timer.refresh()
            self.epoch += 1

        scheduler.step()
        with open(self.log, 'a') as f:
            f.write(self.r_format.format(bid, 0.0, sum(ls) / len(ls), sum(ps) / len(ps)))
        print(self.splitter)

    def test_step(self, data_loader, loss_fn):
        self.eval()
        ls = list()
        with torch.no_grad():
            for bid, batch in enumerate(data_loader):
                hr, lr = batch['hr'].cuda(), batch['lr'].cuda()
                sr = self.forward(lr)
                ls.append(loss_fn(hr, sr).detach().cpu().numpy())
        return np.mean(ls)
