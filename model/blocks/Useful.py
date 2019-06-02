import torch
from torch import nn
import random


class OutputImage(nn.Module):
    def forward(self, x):
        return torch.clamp(torch.round(x), 0., 255.).type('torch.cuda.ByteTensor')


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), rgb_range=255, sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ChannelGradientShuffle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        channels = [0, 1, 2]
        random.shuffle(channels)
        grad_input = grad_input[:, channels, :, :]
        return grad_input
