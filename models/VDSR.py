import torch
import torch.nn as nn
from models.blocks import GeneralBlock

class VDSR(nn.Module):
    def __init__(self, num_blocks=30):
        super(VDSR, self).__init__()
        residual_layers = tuple([GeneralBlock(channel=128, res=False, se=False) for i in range(num_blocks)])
        self.residual_block = nn.Sequential(*residual_layers)
        self.input_conv = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.output_conv = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
        self.activate = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, x):
        tensor = x
        tensor = self.output_conv(self.residual_block(self.activate(self.input_conv(x))))
        return torch.add(x, tensor)
