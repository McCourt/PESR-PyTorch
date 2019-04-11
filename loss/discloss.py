import torch
import torch.nn as nn
import torch.nn.functional as F
from model.discriminator.Discriminator_VGG import Discriminator_VGG_128


class GanLoss(nn.Module):
    def __init__(self, ckpt='checkpoints/vgg_discriminator.ckpt'):
        super().__init__()
        self.discriminator = Discriminator_VGG_128()
        self.discriminator.load_state_dict(torch.load(ckpt))

    def forward(self, sr):
        _, c, h, w = sr.size()
        pad_h, pad_w = 128 - h % 128, 128 - w % 128
        sr = F.pad(sr, (0, pad_w, 0, pad_h), 'constant').permute(0, 2, 3, 1).view((-1, 128, 128, 3)).permute(0, 3, 1, 2)
        return -torch.mean(self.discriminator(sr))
