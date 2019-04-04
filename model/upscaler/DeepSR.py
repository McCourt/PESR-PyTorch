import torch
from torch import nn
from model.blocks import ChannelAttentionBlock, MeanShift, CascadingBlock, ConvolutionBlock, PixelShuffleUpscale


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


class DeepSR(nn.Module):
    def __init__(self, num_groups=None, num_channel=128):
        super().__init__()
        if num_groups is None:
            num_groups = [7, 3]
        self.model_0 = nn.Sequential(
            MeanShift(sign=-1),
            ConvolutionBlock(in_channels=3, out_channels=num_channel)
        )
        self.model_1 = nn.Sequential(*tuple([BasicGroup(num_channel) for _ in range(num_groups[0])]))
        self.upscale_1 = PixelShuffleUpscale(channels=num_channel)

        self.model_2 = nn.Sequential(*tuple([BasicGroup(num_channel) for _ in range(num_groups[1])]))
        self.upscale_2 = PixelShuffleUpscale(channels=num_channel)

        self.model_3 = nn.Sequential(
            ConvolutionBlock(in_channels=num_channel, out_channels=3),
            MeanShift(sign=1)
        )

    def forward(self, x, clip_bound=False):
        x = self.model_0(x)
        up_1 = self.upscale_1(self.model_1(x) + x)
        up_2 = self.upscale_2(self.model_2(up_1) + up_1)
        output = self.model_3(up_2)
        if clip_bound:
            return torch.clamp(torch.round(output), 0., 255.).type('torch.cuda.ByteTensor')
        else:
            return output
