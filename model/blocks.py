import torch
from torch import nn
from torch.nn import functional as F
from math import log2


class OutputImage(nn.Module):
    def forward(self, x):
        return torch.clamp(torch.round(x), 0., 255.).type('torch.cuda.ByteTensor')


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, bias=True,
                 convolution=nn.Conv2d, activation=None, batch_norm=None, padding=1):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        model_body = [
            convolution(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=padding
            )
        ]
        if batch_norm is not None:
            model_body.append(batch_norm(out_channels))
        if activation is not None:
            model_body.append(activation(inplace=True))
        self.model = nn.Sequential(*tuple(model_body))

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, num_blocks=2, res_scale=0.1,
                 activation=nn.LeakyReLU, padding=1, batch_norm=None, bias=True):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        model_body = [
            ConvolutionBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                padding=padding,
                batch_norm=batch_norm,
                bias=bias
            )
        ]
        model_body = model_body + [
            ConvolutionBlock(
                in_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                padding=padding,
                batch_norm=batch_norm,
                bias=bias)
            for _ in range(num_blocks - 2)
        ]
        model_body.append(
            ConvolutionBlock(
                in_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                batch_norm=batch_norm,
                bias=bias
            )
        )

        self.model = nn.Sequential(*tuple(model_body))
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.model(x).mul(self.res_scale)


class CascadingBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, basic_block=ResBlock):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.b1 = basic_block(in_channels, out_channels)
        self.b2 = basic_block(out_channels, out_channels)
        self.b3 = basic_block(out_channels, out_channels)

        self.c1 = basic_block(out_channels * 2, out_channels, kernel_size=1, padding=0)
        self.c2 = basic_block(out_channels * 3, out_channels, kernel_size=1, padding=0)
        self.c3 = basic_block(out_channels * 4, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = torch.cat([x, self.b1(x)], dim=1)
        x = torch.cat([x, self.b2(self.c1(x))], dim=1)
        x = torch.cat([x, self.b3(self.c2(x))], dim=1)
        x = self.c3(x)
        print(x.size())
        return x


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


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), rgb_range=255, sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class PixelShuffleUpscale(nn.Module):
    def __init__(self, channels, scale=2, activation=None, batch_norm=None):
        super().__init__()
        model_body = []
        for _ in range(int(log2(scale))):
            model_body.append(ConvolutionBlock(channels, channels * 4))
            model_body.append(nn.PixelShuffle(2))
            if activation is not None:
                model_body.append(activation(True))
            if batch_norm is not None:
                model_body.append(batch_norm(channels))
        self.model = nn.Sequential(*tuple(model_body))

    def forward(self, x):
        return self.model(x)


class TransposeUpscale(nn.Module):
    def __init__(self, channels, scale=2, activation=nn.LeakyReLU, batch_norm=None):
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
        self.model = nn.Sequential(*tuple(model_body))

    def forward(self, x):
        return self.model(x)


class ConvolutionDownscale(nn.Module):
    def __init__(self, channels, out_channels=None, scale=2):
        super().__init__()
        out_channels = channels if out_channels is None else out_channels

        filter_size = scale * 4
        stride = scale

        pad = scale * 3
        pad_top, pad_left = pad // 2, pad // 2
        pad_bottom, pad_right = pad - pad_top, pad - pad_left

        self.model = nn.Sequential(
            nn.ReplicationPad2d((pad_left, pad_right, pad_top, pad_bottom)),
            nn.Conv2d(in_channels=channels,
                      out_channels=out_channels,
                      kernel_size=filter_size,
                      stride=stride),
            nn.ReLU(inplace=True),
            ConvolutionBlock(in_channels=channels)
        )

    def forward(self, x):
        return self.model(x)
