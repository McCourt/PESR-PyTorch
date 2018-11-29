import torch
from torch import nn
from torch.nn import functional as F
from models.blocks import MeanShift, GeneralBlock, PSUpsample, TransposeUpsample

class EDSR(nn.Module):
    def __init__(self, block_sequence=[15,10,5], num_channel=256,
                block=GeneralBlock, rgb_mean = (0.4488, 0.4371, 0.4040),
                rgb_std = (1.0, 1.0, 1.0), rgb_range = 255):
        super().__init__()

        sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std, sign=-1)
        self.sub_mean = sub_mean
        init = nn.Conv2d(in_channels=3, out_channels=num_channel, kernel_size=3, stride=1, padding=1)
        self.main = nn.Sequential(sub_mean, init)
        seq_1 = [block(channel=num_channel, se=False) for _ in range(block_sequence[0])]
        seq_2 = [block(channel=num_channel, se=False) for _ in range(block_sequence[1])]
        seq_3 = [block(channel=num_channel, se=False) for _ in range(block_sequence[2])]
        seq = []
        seq.extend(seq_1)
        seq.append(PSUpsample(channel=num_channel))
        seq.extend(seq_2)
        seq.append(PSUpsample(channel=num_channel))
        seq.extend(seq_3)
        self.blocks = nn.Sequential(*tuple(seq))

        self.to_3_channel = nn.Conv2d(in_channels=num_channel, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, sign=1)
        self.base = TransposeUpsample(channel=num_channel, out_channel=3)

    def forward(self, x):
        x = self.main(x)
        return self.add_mean(self.to_3_channel(self.blocks(x)) + self.base(x))
