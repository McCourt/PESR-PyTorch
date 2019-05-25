import torch.nn as nn
from model.downscaler.bicubic import BicubicDownSample


class DownScaleLoss(nn.Module):
    def __init__(self, weight=0.5, clip_round=False):
        super().__init__()
        self.down_sampler = BicubicDownSample()
        self.clip_round = clip_round
        self.metric = nn.L1Loss()
        self.w = weight

    def forward(self, hr, sr, lr):
        dsr = self.down_sampler(sr, clip_round=self.clip_round)
        # dhr = self.down_sampler(hr, clip_round=self.clip_round)
        return self.metric(dsr, lr) * self.w + self.metric(sr, hr)
        # return self.metric(dsr, dhr) * self.w + self.metric(sr, hr)
        # return torch.mean(torch.clamp(torch.abs(self.down_sampler(sr, clip_round=self.clip_round) - lr)[:, :, 2:-2, 2:-2], min=self.range))
