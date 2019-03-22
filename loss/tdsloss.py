import torch
import torch.nn as nn
from model.downscaler.TDS import DownSampler


class TrainedDownScaleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_sampler = DownSampler()
        self.down_sampler.load_state_dict(torch.load('/usr/project/xtmp/superresoluter/superresolution/checkpoints/downsampler.pth'))
        self.down_sampler.require_grad = False
        self.require_grad = False
        self.mse = nn.L1Loss()

    def forward(self, sr, lr):
        return self.mse(self.down_sampler(sr), lr)
