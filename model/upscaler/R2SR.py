import torch
import torch.nn as nn
from model.blocks import ConvolutionBlock, Res2Block, TransposeUpscale


class Res2NetSR(nn.Module):
    def __init__(self, num_blocks=32, num_channels=256):
        super().__init__()
        self.conv_in = ConvolutionBlock(in_channels=3, out_channels=num_channels, kernel_size=1, padding=0)
        res_blocks = [Res2Block(in_channels=num_channels, up_channels=num_channels, out_channels=num_channels) for i in range(num_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        self.upsample = TransposeUpscale(channels=num_channels*2, scale=4, mode=2)
        self.conv_out = ConvolutionBlock(in_channels=num_channels*2, out_channels=3)

    def forward(self, x):
        mean = torch.mean(x, dim=(2,3), keepdim=True)
        mean_centered = x - mean
        channel_up = self.conv_in(mean_centered)
        res_out = self.res_blocks(channel_up)
        pre_upsample = torch.cat([res_out, channel_up], dim=1)
        upsample_out = self.upsample(pre_upsample)
        out = self.conv_out(upsample_out) + mean
        return out
