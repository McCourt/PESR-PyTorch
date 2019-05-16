import torch
import torch.nn as nn
from model.downscaler.bicubic import BicubicDownSample


class DownScaleLoss(nn.Module):
    def __init__(self, rg=0.5, clip_round=False):
        super().__init__()
        self.down_sampler = BicubicDownSample()
        self.clip_round = clip_round
        self.mse = nn.L1Loss()
        self.range = rg

    def forward(self, sr, lr):
        # return self.mse(self.down_sampler(sr, clip_round=self.clip_round), lr)
        return torch.mean(torch.clamp(torch.abs(self.down_sampler(sr, clip_round=self.clip_round) - lr)[:, :, 2:-2, 2:-2], min=self.range))
