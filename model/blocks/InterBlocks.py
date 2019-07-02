import torch
from torch import nn


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, bias=True,
                 convolution=nn.Conv2d, activation=None, batch_norm=None, padding=1, rep_pad=False):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        model_body = []
        if rep_pad:
            model_body.append(nn.ReplicationPad2d(padding))
            padding = 0
        model_body.append(
            convolution(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=padding
            )
        )
        if batch_norm is not None:
            model_body.append(batch_norm(out_channels))
        if activation is not None:
            model_body.append(activation(inplace=True))
        self.model = nn.Sequential(*tuple(model_body))

    def forward(self, x):
        return self.model(x)


class DepthSeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, bias=True,
                 convolution=nn.Conv2d, activation=None, batch_norm=None, padding=1, rep_pad=False):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        model_body = []
        if rep_pad:
            model_body.append(nn.ReplicationPad2d(padding))
            padding = 0
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

    
class ShuffleConvBlock(nn.Module):
    """
    Reference: https://github.com/jaxony/ShuffleNet/blob/master/model.py
    """
    def __init__(self, in_channels, out_channels=None, groups=4, kernel_size=3, bias=True,
                 convolution=nn.Conv2d, activation=None, batch_norm=None, padding=1):
        super().__init__()
        self.groups = groups
        self.activation = activation
        out_channels = in_channels if out_channels is None else out_channels
        bottleneck_channels = in_channels//4
        postproc_body = []
        self.preproc = convolution(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                bias=bias,
                groups=groups
        )
        if batch_norm is not None:
            postproc_body.append(batch_norm(bottleneck_channels))
        if activation is not None:
            postproc_body.append(activation(inplace=True))
        postproc_body.append(
            convolution(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=padding,
                groups=bottleneck_channels
            )
        )
        if batch_norm is not None:
            postproc_body.append(batch_norm(bottleneck_channels))
        postproc_body.append(
            convolution(
                in_channels=bottleneck_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=bias,
                groups=groups
            )
        )
        if batch_norm is not None:
            postproc_body.append(batch_norm(out_channels))
        self.postproc = nn.Sequential(*tuple(postproc_body))
        if activation is not None:
            self.activate = activation(inplace=True)
    
    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        residual = x
        x = self.preproc(x)
        x = self.channel_shuffle(x)
        x = self.postproc(x)
        out = residual + x
        if self.activation is not None:
            out = self.activate(out)
        return out
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, num_blocks=2, res_scale=0.1, rep_pad=False,
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
                bias=bias,
                rep_pad=rep_pad
            )
        ]
        model_body = model_body + [
            basic_block(
                in_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                padding=padding,
                batch_norm=batch_norm,
                bias=bias,
                rep_pad=rep_pad
            ) for _ in range(num_blocks - 2)
        ]
        model_body.append(
            basic_block(
                in_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                batch_norm=batch_norm,
                bias=bias,
                rep_pad=rep_pad
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
    def __init__(self, in_channels, out_channels=None, basic_block=ConvolutionBlock, rep_pad=False):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.b1 = ResBlock(in_channels, out_channels, basic_block=basic_block, rep_pad=rep_pad)
        self.b2 = ResBlock(out_channels, out_channels, basic_block=basic_block, rep_pad=rep_pad)
        self.b3 = ResBlock(out_channels, out_channels, basic_block=basic_block, rep_pad=rep_pad)

        self.c1 = ConvolutionBlock(out_channels * 2, out_channels, kernel_size=1, padding=0, rep_pad=rep_pad)
        self.c2 = ConvolutionBlock(out_channels * 3, out_channels, kernel_size=1, padding=0, rep_pad=rep_pad)
        self.c3 = ConvolutionBlock(out_channels * 4, out_channels, kernel_size=1, padding=0, rep_pad=rep_pad)

    def forward(self, x):
        x = torch.cat([x, self.b1(x)], dim=1)
        x = torch.cat([x, self.b2(self.c1(x))], dim=1)
        x = torch.cat([x, self.b3(self.c2(x))], dim=1)
        x = self.c3(x)
        return x
