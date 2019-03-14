import torch
from torch import nn
from model.blocks import MeanShift, ResBlock, ConvolutionBlock, PixelShuffleUpscale, TransposeUpscale


class EDSR(nn.Module):
    def __init__(self, num_blocks=None, num_channel=128, block=ResBlock):
        super().__init__()
        if num_blocks is None:
            num_blocks = [20, 12]
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
        self.model_2 = TransposeUpscale(channels=3, scale=4)
        self.model_3 = nn.Sequential(
            ConvolutionBlock(in_channels=2 * num_channel, out_channels=3),
            MeanShift(sign=1)
        )

    def forward(self, x, clip_bound=False):
        up_x = self.model_2(x)
        init_x = self.model_0(x)
        output = self.model_3(init_x + self.model_1(init_x)) + up_x
        if clip_bound:
            return torch.clamp(torch.round(output), 0., 255.).type('torch.cuda.ByteTensor')
        else:
            return output
