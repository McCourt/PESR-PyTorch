import random
import torch
from torch import nn
from time import time
import json


def since(begin):
    return time() - begin


class Timer(object):
    def __init__(self):
        self.begin = time()

    def report(self):
        return since(self.begin)


def mse_psnr(mse_loss, r=255):
    return 10 * torch.log10(r ** 2 / mse_loss)


class MultiLoss(nn.Module):
    def __init__(self, loss_lst=None):
        super().__init__()
        if loss_lst is None:
            self.loss_lst = {'MSE': 1}
        else:
            self.loss_lst = loss_lst

        if 'MSE' in self.losses.keys():
            self.loss_lst['MSE'] = nn.MSELoss(reduction='elementwise_mean')
        if 'MAE' in self.losses.keys():
            self.loss_lst['MAE'] = nn.L1Loss()
        if 'SmoothMAE' in self.losses.keys():
            self.loss_lst['SmoothMAE'] = nn.SmoothL1Loss()

    def forward(self, sr, hr, lr=None, dsr=None):
        return sum([loss(sr, hr) for loss in self.loss_lst.items()])


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


def load_model(model_name):
    if model_name.lower() == 'srresnet':
        from model.upscaler.SRResNet import SRResNet
        model = SRResNet()
    elif model_name.lower() == 'vdsr':
        from model.upscaler.VDSR import VDSR
        model = VDSR()
    elif model_name.lower() == 'sesr':
        from model.upscaler.SESR import SESR
        model = SESR()
    elif model_name.lower() == 'deepsr':
        from model.upscaler.Model import Model
        model = Model()
    elif model_name.lower() == 'edsr_pyr':
        from model.upscaler.PEDSR import EDSRPyramid
        model = EDSRPyramid()
    elif model_name.lower() == 'deepds':
        from model.downscaler.vgg_ds import DeepDownScale
        model = DeepDownScale()
    elif model_name.lower() == 'res2sr':
        from model.upscaler.Res2NetSR import Res2NetSR
        model = Res2NetSR()
    return model


def report(epoch, bid, l, epoch_loss, bpsnr, epsnr, time):
    return '#E {} | #B {} | Bmse {:6f} | Emse {:6f} | Bpsnr {:6f} | Epsnr {:6f} | RT {:6f}'.format(epoch, bid, l,
                                                                                                   epoch_loss, bpsnr,
                                                                                                   epsnr, time)


def load_parameters(path='./parameter.json'):
    with open(path, 'r') as f:
        return json.load(f)


class ChannelGradientShuffle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        channels = [0, 1, 2]
        random.shuffle(channels)
        grad_input = grad_input[:, channels, :, :]
        return grad_input

