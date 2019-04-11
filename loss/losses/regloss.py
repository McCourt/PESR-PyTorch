import torch.nn as nn


class RegularizationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, sr, sr_tto):
        return self.mse(sr, sr_tto)
