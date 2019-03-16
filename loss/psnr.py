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
            hr = torch.round(hr[:, 0, :, :] * 65.481 + hr[:, 1, :, :] * 128.553 + hr[:, 2, :, :] * 24.966 + 16)
            sr = torch.clamp(torch.round(sr), 0., 255.)
            sr = torch.round(sr[:, 0, :, :] * 65.481 + sr[:, 1, :, :] * 128.553 + sr[:, 2, :, :] * 24.966 + 16)
        return mse_psnr(self.mse(hr, sr))
