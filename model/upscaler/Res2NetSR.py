import torch
import torch.nn as nn
from model.blocks import ConvolutionBlock, Res2Block, TransposeUpscale


class Model(nn.Module):
    def __init__(self, num_blocks=32, num_channels=64):
        super().__init__()
        self.conv_in = ConvolutionBlock(in_channels=3, out_channels=num_channels, kernel_size=1, padding=0)
        res_blocks = [Res2Block(in_channels=num_channels, up_channels=num_channels, out_channels=num_channels) for i in range(num_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        self.upsample = TransposeUpscale(channels=num_channels, scale=4, mode=2)
        self.conv_out = ConvolutionBlock(in_channels=num_channels, out_channels=3)

    def forward(self, x):
        mean = torch.mean(x, dim=(2,3))
        mean_centered = x - mean
        channel_up = self.conv_in(mean_centered)
        in_tensor = self.conv_in(channel_up)
        res_out = self.res_blocks(in_tensor)
        upsample_out = self.upsample(res_out)
        out = self.conv_out(upsample_out) + mean
        return out
