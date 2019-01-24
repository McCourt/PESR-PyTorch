import torch
from torch import nn
from torch.nn import functional as F


def get_activation(activation='relu', inplace=True):
    activation = activation.lower()
    if activation == 'relu':
        return nn.ReLU(inplace=inplace)
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(inplace=inplace)
    else:
        raise NotImplementedError('Activation function {} needs implementation.'.format(activation))


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


class AttentionBlock(nn.Module):
    def __init__(self, channel=256, res=True, se=True,
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
        self.attention = nn.Sigmoid()
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
        residual = self.block(x)
        attention = self.attention(torch.abs(residual))
        residual = residual * attention
        if self.se:
            residual = residual * self.se(self.global_avg(residual).view(n, c)).view(n, c, 1, 1)
        return x + residual if self.res else residual


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', padding='same', bias=True):
        super().__init__()
        ka, kb = 0, 0
        if padding == 'same':
            ka = kernel_size // 2
            kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.model = nn.Sequential(
            nn.ReflectionPad2d((ka, kb, ka, kb)),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      bias=bias),
            get_activation(activation)
        )

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2, res_scale=0.2, kernel_size=3,
                 activation='relu', padding='same', bias=True):
        super().__init__()
        blocks = [ConvolutionBlock(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   activation=activation,
                                   padding=padding, bias=bias)]
        for _ in range(num_blocks - 1):
            blocks.append(
                ConvolutionBlock(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 activation=activation,
                                 padding=padding,
                                 bias=bias)
            )
        self.model = nn.Sequential(*tuple(blocks))
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.block(x).mul(self.res_scale)


class CascadingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=5, res_scale=0.2, kernel_size=3,
                 activation='relu', padding='same', bias=True):
        super().__init__()
        self.blocks = [
            ConvolutionBlock(in_channels=in_channels + cascade * out_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             activation=activation,
                             padding=padding,
                             bias=bias) for cascade in range(num_blocks)
        ]
        self.res_scale = res_scale

    def forward(self, x):
        temp = x
        for block in self.blocks:
            temp = torch.cat((temp, block(temp)), 1)
        return x + temp.mul(self.res_scale)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class TransposeUpsample(nn.Module):
    def __init__(self, channel=64, out_channel=3):
        super().__init__()
        self.process = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        return self.process(x)


class PSUpsample(nn.Module):
    def __init__(self, scale=2, channel=64, activation=False):
        super().__init__()
        self.process = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel * scale ** 2, kernel_size=3, stride=1, padding=1,
                      bias=True),
            nn.PixelShuffle(scale)
        )
        self.activation = activation

    def forward(self, x):
        return nn.LeakyReLU(0.2, inplace=True)(self.process(x)) if self.activation else self.process(x)
