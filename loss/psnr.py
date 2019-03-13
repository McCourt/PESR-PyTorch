import torch
import torch.nn as nn


class PSNR(nn.Module):
    def __init__(self, r=255):
        super().__init__()
        self.r = r
        self.mse = nn.MSELoss()
        self.require_grad = False

    def forward(self, hr, sr):
        return 10 * torch.log10(self.r ** 2 / (self.mse(hr, sr)))
