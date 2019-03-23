import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lap_kernel = torch.Tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]).unsqueeze(0)
        self.loss = nn.L1Loss()
        self.requires_grad = False

    def forward(self, x, y):
        assert x.size() == y.size()
        x_lap = F.conv2d(x, weight=self.lap_kernel, padding=1, groups=x.size()[1])
        y_lap = F.conv2d(y, weight=self.lap_kernel, padding=1, groups=y.size()[1])
        return self.loss(x_lap, y_lap)
