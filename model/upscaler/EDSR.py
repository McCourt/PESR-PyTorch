import torch
from torch import nn
from model.blocks import MeanShift, ResBlock, ConvolutionBlock, PixelShuffleUpscale, TransposeUpscale


class EDSR(nn.Module):
    def __init__(self, num_blocks=[20, 12], num_channel=128, block=ResBlock):
        super().__init__()
        self.model_0 = nn.Sequential(
            MeanShift(sign=-1),
            ConvolutionBlock(in_channels=3, out_channels=num_channel) 
        )
        self.model_1 = nn.Sequential(
            *tuple([block(in_channels=num_channel) for _ in range(num_blocks[0])]),
            TransposeUpscale(channels=num_channel),
            *tuple([block(in_channels=num_channel) for _ in range(num_blocks[1])]),
            TransposeUpscale(channels=num_channel)
        )
        self.model_2 = nn.Sequential(
            TransposeUpscale(channels=num_channel),
            TransposeUpscale(channels=num_channel)
        )
        self.model_3 = nn.Sequential(
            ConvolutionBlock(in_channels=2 * num_channel, out_channels=3),
            MeanShift(sign=1)
        )

    def forward(self, x, clip_bound=False):
        x = self.model_0(x)
        output = self.model_3(torch.cat([self.model_2(x), self.model_1(x)], dim=1))
        if clip_bound:
            return torch.clamp(torch.round(output), 0., 255.).type('torch.cuda.ByteTensor')
        else:
            return output
