import torch
import torch.nn as nn
from src.helper import mse_psnr


class PSNR(nn.Module):
    def __init__(self, r=255):
        super().__init__()
        self.r = r
        self.mse = nn.MSELoss()
        self.require_grad = False

    def forward(self, hr, sr, round_clip=True):
        if round_clip:
            hr = torch.clamp(torch.round(hr), 0., 255.)
            sr = torch.clamp(torch.round(sr), 0., 255.)
        return mse_psnr(self.mse(hr, sr))
