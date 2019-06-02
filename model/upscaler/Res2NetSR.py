import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Average pooling by default
class ChannelAttentionBlock(nn.Module):
    def __init__(self, channel, reduction_ratio=16, pooling=nn.AdaptiveAvgPool2d, activation=nn.ReLU):
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

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat([torch.max(x, dim=1, keepdim=True)[0],
                          torch.mean(x, dim=1, keepdim=True),
                          torch.min(x, dim=1, keepdim=True)[0]], dim=1)

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

class TransposeUpscale(nn.Module):
    def __init__(self, channels, scale=2, activation=nn.LeakyReLU, batch_norm=None):
        super().__init__()
        model_body = []
        for _ in range(int(log2(scale))):
            model_body.append(ConvolutionBlock(channels))
            model_body.append(
                nn.ConvTranspose2d(in_channels=channels,
                                   out_channels=channels,
                                   kernel_size=6,
                                   padding=2,
                                   stride=2)
            )
            if activation is not None:
                model_body.append(activation(True))
            if batch_norm is not None:
                model_body.append(batch_norm(channels))
        self.model = nn.Sequential(*tuple(model_body))

    def forward(self, x):
        return self.model(x)

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

class ResBlock(nn.Module):
    def __init__(self, in_channels=64, up_channels=64, out_channels=64, residual_weight=0.1):
        super().__init__()
        self.residual_weight = residual_weight
        self.conv_1x1_in = ConvolutionBlock(in_channels=in_channels, out_channels=up_channels, kernel_size=1, padding=0, activation=None)
        self.conv_chunk_1 = ConvolutionBlock(in_channels=in_channels//4, out_channels=up_channels//4, activation=nn.LeakyReLU)
        self.conv_chunk_2 = ConvolutionBlock(in_channels=in_channels//4, out_channels=up_channels//4, activation=nn.LeakyReLU)
        self.conv_chunk_3 = ConvolutionBlock(in_channels=in_channels//4, out_channels=up_channels//4, activation=nn.LeakyReLU)
        self.conv_chunk_4 = ConvolutionBlock(in_channels=in_channels//4, out_channels=up_channels//4, activation=nn.LeakyReLU)
        self.conv_1x1_out = ConvolutionBlock(in_channels=up_channels+in_channels, out_channels=out_channels, activation=nn.LeakyReLU)

    def forward(self, x):
        in_tensor = self.conv_1x1_in(x)
        chunks = torch.chunk(in_tensor, chunks=4, dim=1)
        x1 = self.conv_chunk_1(chunks[0])
        x2 = self.conv_chunk_2(chunks[1] + x1 * self.residual_weight)
        x3 = self.conv_chunk_3(chunks[2] + x2 * self.residual_weight)
        x4 = self.conv_chunk_4(chunks[3] + x3 * self.residual_weight)
        concat = torch.cat([x, x1, x2, x3, x4], dim=1) # Maybe add a decay rate to deal with potential numerical stability
        out = self.conv_1x1_out(concat)
        return out

class Res2NetSR(nn.Module):
    def __init__(self, num_blocks=32, num_channels=64):
        super().__init__()
        self.conv_in = ConvolutionBlock(in_channels=3, out_channels=num_channels, kernel_size=1, padding=0)
        res_blocks = [ResBlock(in_channels=num_channels, up_channels=num_channels, out_channels=num_channels) for i in range(num_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        self.upsample = TransposeUpscale(channels=num_channels, scale=4)
        self.conv_out = ConvolutionBlock(in_channels=num_channels, out_channels=3)

    def forward(self, x):
        mean = torch.mean(x, dim=(2,3))
        mean_centered = x - mean
        channel_up = self.conv_in(mean_centered)
        in_tensor = self.conv_in(channel_up)
        res_out = self.res_blocks(in_tensor)
        upsample_out = self.upsample(res_out)
        out = self.conv_out(upsample_out) + mean
        return out
        

        
