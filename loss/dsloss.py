import torch.nn as nn
from model.downscaler.bicubic import BicubicDownSample


class DownsampleLoss(nn.Module):
    def __init__(self, scale, clip_round=False, metric=nn.L1Loss, downsampler=BicubicDownSample):
        super().__init__()
        self.down_sampler = downsampler(factor=scale)
        self.clip_round = clip_round
        self.metric = metric()

    def forward(self, sr, lr):
        dsr = self.down_sampler(sr, clip_round=self.clip_round)
        dhr = self.down_sampler(hr, clip_round=self.clip_round)
        return self.metric(dsr, lr)


class DSLoss(nn.Module):
    def __init__(self, scale, weight=0.02, clip_round=False, metric=nn.L1Loss, decay_ratio=0.9977):
        super().__init__()
        self.downsample_loss = DownsampleLoss(scale=factor, clip_round=clip_round, metric=metric)
        self.clip_round = clip_round
        self.metric = metric()
        self.w = weight
        self.p = decay_ratio

    def decay(self):
        self.w *= self.p
        
    def forward(self, hr, sr, lr):
        return self.downsample_loss(sr, lr) * self.w + self.metric(sr, hr)