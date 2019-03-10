import torch
from torch.nn import functional as F

class ShiftLoss(torch.nn.Module):
    def __init__(self):
        self.kernel_dict = torch.load('shifts10.pt')
        super(ShiftLoss, self).__init__()

    def forward(self, lr, hr):
        loss = 0
        for i in range(-2, 2):
            for j in range(-2, 2):
                x = F.conv2d(input=lr, weight=self.kernel_dict[(i, j)], stride=1)
                y = hr[:, :, 10 + i:-10 + i, 10 + j:-10 + j]
                loss += F.mse_loss(x, y)
        loss = loss / 16
        return loss