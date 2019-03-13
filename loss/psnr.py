import torch.nn as nn
from src.helper import psnr


class PSNR(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.require_grad = False

    def forward(self, hr, sr):
        return psnr(self.mse(hr, sr))
