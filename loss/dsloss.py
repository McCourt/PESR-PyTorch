import torch.nn as nn
from model.downscaler.bicubic import BicubicDownSample


class DownScaleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_sampler = BicubicDownSample()
        self.mse = nn.L1Loss()

    def forward(self, sr, lr):
        return self.mse(self.down_sampler(sr), lr)
