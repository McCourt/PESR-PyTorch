from torch import nn
from model.blocks import ResBlock, PixelShuffleUpscale, ConvolutionBlock


class SRResNet(nn.Module):
    def __init__(self, num_res=30, channel=128):
        super().__init__()
        self.model = nn.Sequential(
            ConvolutionBlock(in_channels=3, out_channels=channel),
            *tuple([ResBlock(in_channels=channel) for _ in range(num_res)]),
            ConvolutionBlock(in_channels=channel),
            PixelShuffleUpscale(channels=channel),
            PixelShuffleUpscale(channels=channel),
            ConvolutionBlock(in_channels=channel, out_channels=3)
        )

    def forward(self, x):
        return self.model(x) + x
