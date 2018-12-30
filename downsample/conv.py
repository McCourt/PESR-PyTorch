import torch
import torch.nn as nn
import torch.functional as F


class ConvolutionDownscale(nn.Module):
    def __init__(self, scale=4, channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=scale ** 2,
                      kernel_size=channels,
                      stride=channels,
                      padding=0),
            nn.Conv2d(in_channels=scale ** 2,
                      out_channels=channels,
                      kernel_size=1,
                      stride=1,
                      padding=0)
        )

    def forward(self, x):
        return self.model(x)
