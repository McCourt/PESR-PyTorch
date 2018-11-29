import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from bicubic import BicubicDownSample
from time import time
import os


def since(begin):
    return time() - begin


class Timer(object):
    def __init__(self):
        self.begin = time()

    def report(self):
        return since(self.begin)


def psnr(mse_loss, R=255):
    return 10 * torch.log10(R ** 2 / mse_loss)


class MSEnDSLoss(nn.Module):
    def __init__(self, add_ds=True, ds=None):
        super().__init__()
        self.add_ds = add_ds
        self.hr = nn.MSELoss(reduction='elementwise_mean')
        if self.add_ds:
            self.ds = ds
            self.lr = nn.MSELoss(reduction='elementwise_mean')

    def forward(self, x, hr, lr):
        loss = self.hr(x, hr)
        if self.add_ds and lr is not None:
            ds = self.ds(x)
            loss += self.lr(ds, lr)
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
