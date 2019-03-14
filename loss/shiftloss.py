import torch
import torch.nn as nn
import torch.nn.functional as F
from model.downscaler.bicubic import BicubicDownSample


class ShiftLoss(nn.Module):
    def __init__(self, kernel_dir='checkpoints/shifts.ckpt'):
        super(ShiftLoss, self).__init__()
        self.kernel_dict = torch.load(kernel_dir)
        self.bicubic = BicubicDownSample()

    def forward(self, hr, lr):
        loss = 0
        for i in range(-2, 2):
            for j in range(-2, 2):
                kernel = self.kernel_dict[(i, j)].type('torch.cuda.FloatTensor')
                x = F.conv2d(input=lr, weight=kernel, stride=1)
                y = self.bicubic(hr.roll((-i, -j), (2, 3)))[:, :, 10:-10, 10:-10]
                loss += F.mse_loss(x, y)
        loss = loss / 16
        return loss
