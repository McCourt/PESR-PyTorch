import torch
from torch import nn
from torch.nn import functional as F
from math import log2


class OutputImage(nn.Module):
    def forward(self, x):
        return torch.clamp(torch.round(x), 0., 255.).type('torch.cuda.ByteTensor')


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, bias=True,
                 convolution=nn.Conv2d, activation=nn.ReLU, batch_norm=None, padding=nn.ConstantPad2d):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        model_body = []
        if padding is not None:
            ka = (kernel_size - 1) // 2
            kb = (kernel_size - 1) - ka
            model_body.append(padding((ka, kb, ka, kb), 0))
        model_body.append(
            convolution(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        bias=bias)
        )
        if batch_norm is not None:
            model_body.append(batch_norm(out_channels))
        if activation is not None:
            model_body.append(activation(inplace=True))
        self.model = nn.Sequential(*tuple(model_body))

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, num_blocks=2, res_scale=1,
                 activation=nn.ReLU, padding=nn.ConstantPad2d, batch_norm=None, bias=True):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        model_body = [
            ConvolutionBlock(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             activation=activation,
                             padding=padding,
                             batch_norm=batch_norm,
                             bias=bias)
        ]
        for _ in range(num_blocks - 2):
            model_body.append(
                ConvolutionBlock(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 activation=activation,
                                 padding=padding,
                                 batch_norm=batch_norm,
                                 bias=bias)
            )
        model_body.append(
            ConvolutionBlock(in_channels=out_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             activation=None,
                             padding=padding,
                             batch_norm=batch_norm,
                             bias=bias)
        )

        self.model = nn.Sequential(*tuple(model_body))
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.model(x).mul(self.res_scale)


class CascadingBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, basic_block=ResBlock):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.b1 = ResBlock(in_channels, out_channels)
        self.b2 = ResBlock(out_channels, out_channels)
        self.b3 = ResBlock(out_channels, out_channels)

        self.c1 = ResBlock(out_channels * 2, out_channels, kernel_size=1)
        self.c2 = ResBlock(out_channels * 4, out_channels, kernel_size=1)
        self.c3 = ResBlock(out_channels * 8, out_channels, kernel_size=1)

    def forward(self, x):
        x = torch.cat([x, self.b1(x)], dim=1)
        x = torch.cat([x, self.b2(self.c1(x))], dim=1)
        x = torch.cat([x, self.b3(self.c2(x))], dim=1)
        return self.c3(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.concat([torch.max(x, dim=1, keepdim=True),
                             torch.mean(x, dim=1, keepdim=True),
                             torch.min(x, dim=1, keepdim=True)], dim=1)


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


class MeanShift(nn.Module):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super().__init__()
        std = torch.Tensor(rgb_std)
        mean = torch.Tensor(rgb_mean)
        self.weight = (torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)).type('torch.cuda.FloatTensor')
        self.bias = (sign * rgb_range * mean / std).type('torch.cuda.FloatTensor')
        self.requires_grad = False

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias)


class PixelShuffleUpscale(nn.Module):
    def __init__(self, channels, scale, activation=None, batch_norm=None):
        super().__init__()
        model_body = []
        for _ in range(int(log2(scale))):
            model_body.append(ConvolutionBlock(channels, channels * 2))
            model_body.append(nn.PixelShuffle(2))
            if activation is not None:
                model_body.append(activation(True))
            if batch_norm is not None:
                model_body.append(batch_norm(channels))
        self.model = nn.Sequential(model_body)

    def forward(self, x):
        return self.model(x)


class TransposeUpscale(nn.Module):
    def __init__(self, channels, scale, activation=nn.LeakyReLU, batch_norm=None):
        super().__init__()
        model_body = []
        for _ in range(int(log2(scale))):
            model_body.append(ConvolutionBlock(channels))
            model_body.append(
                nn.ConvTranspose2d(in_channels=channels,
                                   out_channels=channels,
                                   kernel_size=4,
                                   padding=1,
                                   stride=2)
            )
            if activation is not None:
                model_body.append(activation(True))
            if batch_norm is not None:
                model_body.append(batch_norm(channels))
        self.model = nn.Sequential(model_body)

    def forward(self, x):
        return self.model(x)


class GeneralBlock(nn.Module):
    def __init__(self, channel=256, res_scale=0.8, res=True, se=True,
                 bias=True, batch_norm=False, activation=True, num_blocks=2):
        super().__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=bias))
            if activation:
                blocks.append(nn.LeakyReLU(inplace=True, negative_slope=0.1))
            if batch_norm:
                blocks.append(nn.batch_norm(channel))
        self.block = nn.Sequential(*tuple(blocks))
        self.se = se
        self.res = res
        self.res_scale = res_scale
        if self.se:
            self.global_avg = nn.AdaptiveAvgPool2d(1)
            self.se = nn.Sequential(
                nn.Linear(channel, channel),
                nn.ReLU(inplace=True),
                nn.Linear(channel, channel),
                nn.Sigmoid()
            )

    def forward(self, x):
        n, c, h, w = x.size()
        residual = self.res_scale * self.block(x)
        if self.se:
            residual = residual * self.se(self.global_avg(residual).view(n, c)).view(n, c, 1, 1)
        return x + residual if self.res else residual
