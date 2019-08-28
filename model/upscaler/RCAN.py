from torch import nn
from model.blocks import ChannelAttV2, MeanShift
from model.blocks import ResBlock, ConvolutionBlock, PixelShuffleUpscale, DepthSeparableConvBlock
from math import log2


class BasicBlock(nn.Module):
    def __init__(self, num_channel, res_scale=1, attention=ChannelAttV2):
        super().__init__()
        self.model_body = nn.Sequential(
            ResBlock(num_channel),
            ResBlock(num_channel),
            attention(num_channel)
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


class RCAN(nn.Module):
    def __init__(self, scale=4, num_groups=20, num_channel=64):
        super().__init__()
        self.model_0 = nn.Sequential(
            MeanShift(sign=-1),
            ConvolutionBlock(in_channels=3, out_channels=num_channel)
        )
        self.model_1 = nn.Sequential(*tuple([BasicGroup(num_channel) for _ in range(num_groups)]))

        upscaler = [PixelShuffleUpscale(channels=num_channel) for _ in range(int(log2(scale)))]
        self.model_2 = nn.Sequential(
            *tuple(upscaler),
            ConvolutionBlock(in_channels=num_channel, out_channels=3),
            MeanShift(sign=1)
        )

    def forward(self, x):
        o = self.model_0(x)
        o = self.model_1(o) + o
        o = self.model_2(o)
        return o
