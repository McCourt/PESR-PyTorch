import torch
from torch import nn
from model.blocks import ConvolutionBlock
from math import log2


class PixelShuffleUpscale(nn.Module):
    def __init__(self, channels, scale=2, activation=None, batch_norm=None, basic_block=ConvolutionBlock, **kargs):
        super().__init__()
        model_body = []
        for _ in range(int(log2(scale))):
            model_body.append(basic_block(channels, channels * 4, **kargs))
            model_body.append(nn.PixelShuffle(2))
            if activation is not None:
                model_body.append(activation(True))
            if batch_norm is not None:
                model_body.append(batch_norm(channels))
        self.model = nn.Sequential(*tuple(model_body))

    def forward(self, x):
        return self.model(x)


class TransposeUpscale(nn.Module):
    def __init__(self, channels, scale=2, activation=nn.LeakyReLU, batch_norm=None, mode=1, rep_pad=False):
        super().__init__()
        if mode == 1:
            k_size, p_size = 4, 1
        elif mode == 2:
            k_size, p_size = 6, 2
        else:
            raise ValueError('wrong mode parameters')
        model_body = []
        for _ in range(int(log2(scale))):
            model_body.append(ConvolutionBlock(channels))
            model_body.append(
                nn.ConvTranspose2d(in_channels=channels,
                                   out_channels=channels,
                                   kernel_size=k_size,
                                   padding=p_size,
                                   stride=2,
                                   rep_pad=rep_pad)
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
