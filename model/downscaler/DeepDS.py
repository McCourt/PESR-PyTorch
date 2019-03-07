import torch
from torch import nn
from model.blocks import ConvolutionBlock, MeanShift, ResBlock, TransposeUpscale, ConvolutionDownscale


class DeepDownScale(nn.Module):
    def __init__(self, num_blocks=None, num_channel=64, block=ResBlock):
        super().__init__()
        if num_blocks is None:
            num_blocks = [20, 10, 5]
        assert len(num_blocks) == 3
        self.init = nn.Sequential(
            MeanShift(sign=-1),
            ConvolutionBlock(in_channels=3, out_channels=num_channel)
        )

        self.model_1 = nn.Sequential(*tuple([block(in_channels=num_channel) for _ in range(num_blocks[0])]))
        self.model_2 = nn.Sequential(*tuple([block(in_channels=num_channel) for _ in range(num_blocks[1])]))
        self.model_3 = nn.Sequential(*tuple([block(in_channels=num_channel) for _ in range(num_blocks[2])]))

        self.downscale_1 = ConvolutionDownscale(channels=num_channel)
        self.downscale_2 = ConvolutionDownscale(channels=num_channel)
        self.upscale_1 = TransposeUpscale(channels=num_channel)
        self.upscale_2 = TransposeUpscale(channels=2 * num_channel)

        self.end = nn.Sequential(
            ConvolutionBlock(in_channels=num_channel * 3, out_channels=3),
            MeanShift(sign=1)
        )

    def forward(self, x):
        x = self.init(x)
        ds_1 = self.downscale_1(x)
        ds_2 = self.downscale_2(ds_1)

        x_out = torch.cat(
            [
                self.model_1(x), 
                self.upscale_2(
                    torch.cat(
                        [
                            self.model_2(ds_1), 
                            self.upscale_1(self.model_3(ds_2))
                        ], dim=1
                    )
                )
            ], dim=1
        )
        return self.end(x_out)
