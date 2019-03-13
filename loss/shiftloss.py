import torch
import torch.nn as nn
import torch.nn.functional as F
from model.downscaler.bicubic import BicubicDownSample


class ShiftLoss(nn.Module):
    def __init__(self, kernel_dir='checkpoints/shifts.ckpt'):
        super(ShiftLoss, self).__init__()
        self.kernel_dict = torch.load(kernel_dir)
        self.bicubic = BicubicDownSample()
        for key in self.kernel_dict.keys():
            self.kernel_dict[key] = self.kernel_dict[key].unsqueeze_(0)

    def forward(self, hr, lr):
        loss = 0
        for i in range(-2, 2):
            for j in range(-2, 2):
                kernel = self.kernel_dict[(i, j)].type('torch.cuda.FloatTensor')
                x = F.conv2d(input=lr, weight=kernel, stride=1, padding=3)
                y = self.bicubic(hr.roll((i, j), (2, 3)))
                loss += F.mse_loss(x, y)
        loss = loss / 16
        return loss
