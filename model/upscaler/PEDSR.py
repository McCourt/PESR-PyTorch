from torch import nn
from model.blocks import MeanShift, ResBlock, PixelShuffleUpscale, TransposeUpscale, ConvolutionBlock


class EDSRPyramid(nn.Module):
    def __init__(self, block_sequence=None, num_channel=256,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0), rgb_range=255):
        super().__init__()

        if block_sequence is None:
            block_sequence = [15, 10, 5]

        self.init = nn.Sequential(
            MeanShift(rgb_range, rgb_mean, rgb_std, sign=-1),
            ConvolutionBlock(in_channels=3, out_channels=num_channel)
        )

        self.skip_upscale = nn.Sequential(
            TransposeUpscale(channels=num_channel),
            ConvolutionBlock(in_channels=num_channel, out_channels=3)
        )

        model_body = []
        for seq in block_sequence:
            model_body.extend([ResBlock(in_channels=num_channel) for _ in range(seq)])
            model_body.append(PixelShuffleUpscale(channels=num_channel))
        model_body.append(ConvolutionBlock(in_channels=num_channel, out_channels=3))
        self.blocks = nn.Sequential(*tuple(model_body))

        self.model_end = nn.Sequential(
            ConvolutionBlock(in_channels=num_channel, out_channels=3),
            MeanShift(rgb_range, rgb_mean, rgb_std, sign=1)
        )

    def forward(self, x):
        x = self.init(x)
        return self.model_end(self.skip_upscale(x) + self.blocks(x))
