import random

import torch
from torch import nn
from time import time
import os
import json


def since(begin):
    return time() - begin


class Timer(object):
    def __init__(self):
        self.begin = time()

    def report(self):
        return since(self.begin)


def psnr(mse_loss, r=255):
    return 10 * torch.log10(r ** 2 / mse_loss)


class MSEnDSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.hr_loss = nn.MSELoss(reduction='elementwise_mean')
        self.lr_loss = nn.MSELoss(reduction='elementwise_mean')

    def forward(self, sr, hr, lr, dsr=None, lambda_lr=0.2):
        loss = self.hr_loss(sr, hr)
        if dsr is not None:
            loss += lambda_lr * self.lr_loss(dsr, lr)
        return loss


def save_checkpoint(state_dict, save_dir, model_name='model'):
    try:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(state_dict, os.path.join(save_dir, model_name + '.ckpt'))
        print('checkpoint saved')
    except:
        raise Exception('checkpoint saving failure')


def load_checkpoint(load_dir, model_name='model', map_location=None):
    try:
        if not os.path.exists(load_dir):
            print('No checkpoint and begin new training')
            return None
        else:
            print('loading checkpoint')
            checkpoint = torch.load(os.path.join(load_dir, model_name + '.ckpt'), map_location=map_location)
            print('loading successful')
            return checkpoint
    except:
        print('No checkpoint and begin new training')


def load_model(model_name):
    if model_name.lower() == 'srresnet':
        from models.SRResNet import SRResNet
        model = SRResNet()
    elif model_name.lower() == 'vdsr':
        from models.VDSR import VDSR
        model = VDSR()
    elif model_name.lower() == 'sesr':
        from models.SESR import SESR
        model = SESR()
    elif model_name.lower() == 'edsr':
        from models.EDSR import EDSR
        model = EDSR()
    elif model_name.lower() == 'edsr_pyr':
        from models.EDSR_Pyramid import EDSR
        model = EDSR()
    return model


def report(epoch, bid, l, epoch_loss, bpsnr, epsnr, time):
    return '#E {} | #B {} | Bmse {:6f} | Emse {:6f} | Bpsnr {:6f} | Epsnr {:6f} | RT {:6f}'.format(epoch, bid, l,
                                                                                                   epoch_loss, bpsnr,
                                                                                                   epsnr, time)


def load_parameters(key, path='./parameter.json'):
    with open(path, 'r') as f:
        for pars in json.load(f):
            if pars['method'] == key:
                return pars


class ChannelGradientShuffle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        channels = [1, 2, 3]
        random.shuffle(channels)
        grad_input = grad_input[:, channels, :, :]
        return grad_input

