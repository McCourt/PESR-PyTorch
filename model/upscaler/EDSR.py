import torch
from torch import nn
from model.blocks import ResBlock, MeanShift, ConvolutionBlock, PixelShuffleUpscale
from math import log2


class EDSR(nn.Module):
    def __init__(self, scale, num_groups=32, num_channel=256, rep_pad=False):
        super().__init__()
        self.model_0 = nn.Sequential(
            MeanShift(sign=-1),
            ConvolutionBlock(in_channels=3, out_channels=num_channel) 
        )
        self.model_1 = nn.Sequential(*tuple([ResBlock(num_channel) for _ in range(num_groups)]))
        self.model_2 = nn.Sequential(
            *tuple([PixelShuffleUpscale(channels=num_channel) for _ in range(int(log2(scale)))]),
            ConvolutionBlock(in_channels=num_channel, out_channels=3),
            MeanShift(sign=1)
        )

    def forward(self, x):
        o = self.model_0(x)
        o = self.model_1(o) + o
        o = self.model_2(o)
        return o
