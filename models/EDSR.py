import torch
from torch import nn
from torch.nn import functional as F
from models.blocks import MeanShift, ResBlock, ConvolutionBlock
from helper import weights_init

class EDSR(nn.Module):
    def __init__(self, num_blocks=30, num_channel=128, block=ResBlock,
                rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), rgb_range=255):
        super().__init__()

        self.model_1 = nn.Sequential(
            MeanShift(rgb_range=rgb_range, rgb_mean=rgb_mean, rgb_std=rgb_std, sign=-1),
            ConvolutionBlock(in_channels=3, out_channels=num_channel),
            *tuple([block(in_channels=num_channel) for _ in range(num_blocks)])
        )

        self.model_2 = nn.Sequential(
            ConvolutionBlock(in_channels=num_channel, out_channels=3),
            MeanShift(rgb_range=rgb_range, rgb_mean=rgb_mean, rgb_std=rgb_std, sign=1)
        )

    def forward(self, x, clip_bound=False):
        output = self.model_2(x + self.model_1(x))
        if clip_bound:
            return torch.clamp(torch.round(output), 0., 255.).type('torch.cuda.ByteTensor')
        else:
            return output
