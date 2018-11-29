import torch
from torch import nn
from torch.nn import functional as F
from models.blocks import MeanShift, GeneralBlock, TransposeUpsample

class EDSR(nn.Module):
    def __init__(self, num_blocks=30, num_channel=256, block=GeneralBlock,
                rgb_mean=(0.4488, 0.4371, 0.4040),
                rgb_std=(1.0, 1.0, 1.0), rgb_range=255):
        super().__init__()

        sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std, sign=-1)
        init = nn.Conv2d(in_channels=3, out_channels=num_channel, kernel_size=3, stride=1, padding=1)
        self.main = nn.Sequential(sub_mean, init)

        self.blocks = nn.Sequential(*tuple([GeneralBlock(channel=num_channel, se=False) for _ in range(num_blocks)]))

        upsample = TransposeUpsample(channel=num_channel)
        add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, sign=1)
        self.upsample = nn.Sequential(upsample, add_mean)


    def forward(self, x):
        x = self.main(x)
        # x = torch.round(self.upsample(x + self.blocks(x)))
        # x = torch.clamp(torch.round(self.upsample(x + self.blocks(x))), 0., 255.)
        return self.upsample(x + self.blocks(x))
