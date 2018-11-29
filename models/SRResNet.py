import torch
from torch import nn
from torch.nn import functional as F
from models.blocks import GeneralBlock, PSUpsample

class SRResNet(nn.Module):
    def __init__(self, num_res=30, channel=256):
        super().__init__()
        init = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        res_block = nn.Sequential(
            *tuple([GeneralBlock(channel=channel, se=False) for _ in range(num_res)])
        )

        smooth = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=False)
        )

        upscale4x = nn.Sequential(
            PSUpsample(channel=channel),
            PSUpsample(channel=channel)
        )

        output = nn.Conv2d(in_channels=channel, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)

        self.process = nn.Sequential(
            init,
            res_block,
            smooth,
            upscale4x,
            output
        )

    def forward(self, x):
        x = self.process(x)
        return x
