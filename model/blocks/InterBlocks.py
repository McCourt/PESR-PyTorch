import torch
from torch import nn


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


class DepthSeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, bias=True,
                 convolution=nn.Conv2d, activation=None, batch_norm=None, padding=1):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        model_body = []
        model_body.append(
            convolution(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=padding,
                groups=in_channels
            )
        )
        if batch_norm is not None:
            model_body.append(batch_norm(out_channels))
        if activation is not None:
            model_body.append(activation(inplace=True))
        model_body.append(
            convolution(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=bias
            )
        )
        if batch_norm is not None:
            model_body.append(batch_norm(out_channels))
        if activation is not None:
            model_body.append(activation(inplace=True))
        self.model = nn.Sequential(*tuple(model_body))

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, num_blocks=2, res_scale=0.1,
                 activation=nn.LeakyReLU, padding=1, batch_norm=None, bias=True, basic_block=ConvolutionBlock):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        model_body = [
            basic_block(
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
            basic_block(
                in_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                padding=padding,
                batch_norm=batch_norm,
                bias=bias)
            for _ in range(num_blocks - 2)
        ]
        model_body.append(
            basic_block(
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


class Res2Block(nn.Module):
    def __init__(self, in_channels=64, up_channels=64, out_channels=64, residual_weight=0.5):
        super().__init__()
        self.residual_weight = residual_weight
        self.conv_1x1_in = ConvolutionBlock(in_channels=in_channels, out_channels=up_channels, kernel_size=1, padding=0, activation=None)
        sub_in, sub_out = in_channels // 4, up_channels // 4
        self.conv_chunk_1 = ConvolutionBlock(in_channels=sub_in, out_channels=sub_out, activation=nn.LeakyReLU)
        self.conv_chunk_2 = ConvolutionBlock(in_channels=sub_in, out_channels=sub_out, activation=nn.LeakyReLU)
        self.conv_chunk_3 = ConvolutionBlock(in_channels=sub_in, out_channels=sub_out, activation=nn.LeakyReLU)
        self.conv_chunk_4 = ConvolutionBlock(in_channels=sub_in, out_channels=sub_out, activation=nn.LeakyReLU)
        self.conv_1x1_out = ConvolutionBlock(in_channels=up_channels,
                                             out_channels=out_channels,
                                             activation=nn.LeakyReLU)

    def forward(self, x):
        in_tensor = self.conv_1x1_in(x)
        chunks = torch.chunk(in_tensor, chunks=4, dim=1)
        x1 = self.conv_chunk_1(chunks[0])
        x2 = self.conv_chunk_2(chunks[1] + x1 * self.residual_weight)
        x3 = self.conv_chunk_3(chunks[2] + x2 * self.residual_weight)
        x4 = self.conv_chunk_4(chunks[3] + x3 * self.residual_weight)
        concat = torch.cat([x1, x2, x3, x4], dim=1)
        # Maybe add a decay rate to deal with potential numerical stability
        out = self.conv_1x1_out(concat) + in_tensor
        return out


class CascadingBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, basic_block=ConvolutionBlock):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.b1 = ResBlock(in_channels, out_channels, basic_block=basic_block)
        self.b2 = ResBlock(out_channels, out_channels, basic_block=basic_block)
        self.b3 = ResBlock(out_channels, out_channels, basic_block=basic_block)

        self.c1 = ConvolutionBlock(out_channels * 2, out_channels, kernel_size=1, padding=0)
        self.c2 = ConvolutionBlock(out_channels * 3, out_channels, kernel_size=1, padding=0)
        self.c3 = ConvolutionBlock(out_channels * 4, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = torch.cat([x, self.b1(x)], dim=1)
        x = torch.cat([x, self.b2(self.c1(x))], dim=1)
        x = torch.cat([x, self.b3(self.c2(x))], dim=1)
        x = self.c3(x)
        return x
