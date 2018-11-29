import torch
from torch import nn
from torch.nn import functional as F
from models.blocks import GeneralBlock, PSUpsample

class SESR(nn.Module):
    def __init__(self, num_se=30, channel=256, bias=True):
        super().__init__()
        init = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True)
        )

        se_blocks = nn.Sequential(
            *tuple([GeneralBlock(channel=channel, se=True, bias=bias) for i in range(num_se)])
        )

        smooth = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=bias)

        upscale4x = nn.Sequential(
            PSUpsample(channel=channel),
            PSUpsample(channel=channel)
        )

        output = nn.Conv2d(in_channels=channel, out_channels=3, kernel_size=9, stride=1, padding=4, bias=bias)

        self.process = nn.Sequential(
            init,
            se_blocks,
            smooth,
            upscale4x,
            output
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, x):
        x = self.process(x)
        return x
