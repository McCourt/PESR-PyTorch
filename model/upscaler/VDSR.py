import torch
import torch.nn as nn
from model.blocks import ResBlock, ConvolutionBlock


class Model(nn.Module):
    def __init__(self, num_blocks=50, channel=128):
        super().__init__()
        self.model = nn.Sequential(
            ConvolutionBlock(in_channels=3, out_channels=channel),
            *tuple([ResBlock(in_channels=channel) for _ in range(num_blocks)]),
            ConvolutionBlock(in_channels=channel)
        )

    def forward(self, x):
        return self.model(x) + x
