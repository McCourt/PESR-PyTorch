import torch
from torch import nn
from model.blocks import ChannelAttentionBlock, MeanShift, CascadingBlock, ConvolutionBlock, PixelShuffleUpscale, DepthSeparableConvBlock


class BasicBlock(nn.Module):
    def __init__(self, num_channel, res_scale=1):
        super().__init__()
        self.model_body = nn.Sequential(
            CascadingBlock(num_channel),
            ChannelAttentionBlock(num_channel)
        )
        self.res_scale = res_scale

    def forward(self, x):
        return self.model_body(x) * self.res_scale + x


class BasicGroup(nn.Module):
    def __init__(self, num_channel, num_block=5, res_scale=1):
        super().__init__()
        self.model_body = nn.Sequential(
            *tuple([BasicBlock(num_channel) for _ in range(num_block)]),
            ConvolutionBlock(in_channels=num_channel)
        )
        self.res_scale = res_scale

    def forward(self, x):
        return self.model_body(x) * self.res_scale + x


class EDSR(nn.Module):
    def __init__(self, num_groups=4, num_channel=256):
        super().__init__()
        self.model_0 = nn.Sequential(
            MeanShift(sign=-1),
            ConvolutionBlock(in_channels=3, out_channels=num_channel)
        )
        self.model_1 = nn.Sequential(*tuple([BasicGroup(num_channel) for _ in range(num_groups)]))

        self.model_2 = nn.Sequential(
            PixelShuffleUpscale(channels=num_channel),
            PixelShuffleUpscale(channels=num_channel),
            ConvolutionBlock(in_channels=num_channel, out_channels=3),
            MeanShift(sign=1)
        )

    def forward(self, x):
        o = self.model_0(x)
        o = self.model_1(o) + o
        o = self.model_2(o)
        return o
