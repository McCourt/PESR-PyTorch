import torch.nn as nn
from model.downscaler.bicubic import BicubicDownSample


class DownScaleLoss(nn.Module):
    def __init__(self, scale, weight=0.1, clip_round=False):
        super().__init__()
        self.down_sampler = BicubicDownSample(factor=scale)
        self.clip_round = clip_round
        self.metric = nn.L1Loss()
        self.w = weight
        self.r = 0.5

    def forward(self, hr, sr, lr):
        dsr = self.down_sampler(sr, clip_round=self.clip_round)
        # dhr = self.down_sampler(hr, clip_round=self.clip_round)
        # dsl = torch.mean(torch.clamp(torch.abs(self.down_sampler(sr, clip_round=self.clip_round) - lr)[:, :, 2:-2, 2:-2], min=self.r)
        return self.metric(dsr, lr) * self.w + self.metric(sr, hr)
        # return self.metric(dsr, dhr) * self.w + self.metric(sr, hr)
        # return dsl * self.w + self.metric(sr, hr)
