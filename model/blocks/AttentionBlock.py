import torch
from torch import nn
from model.blocks import ConvolutionBlock


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat([torch.max(x, dim=1, keepdim=True)[0],
                          torch.mean(x, dim=1, keepdim=True),
                          torch.min(x, dim=1, keepdim=True)[0]], dim=1)


class ChannelAttentionBlock(nn.Module):
    def __init__(self, channel, reduction_ratio=16, pooling=nn.AdaptiveMaxPool2d, activation=nn.ReLU):
        super().__init__()
        self.model = nn.Sequential(
            pooling(1),
            Flatten(),
            nn.Linear(channel, channel // reduction_ratio),
            activation(True),
            nn.Linear(channel // reduction_ratio, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.model(x).unsqueeze(2).unsqueeze(3).expand_as(x)


class SpatialAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ChannelPool(),
            ConvolutionBlock(in_channels=3,
                             out_channels=1,
                             activation=None),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.model(x)