import torch
from torch.nn import functional as F
from model.downscaler.bicubic import BicubicDownSample


class ShiftLoss(torch.nn.Module):
    def __init__(self, kernel_dir='checkpoints/shifts.ckpt'):
        super(ShiftLoss, self).__init__()
        self.kernel_dict = torch.load(kernel_dir)
        self.bicubic = BicubicDownSample()

    def forward(self, hr, lr):
        loss = 0
        for i in range(-2, 2):
            for j in range(-2, 2):
                x = F.conv2d(input=lr, weight=self.kernel_dict[(i, j)], stride=1)
                y = self.bicubic(hr.roll(0 ,0 ,i ,j))[:, :, 10:-10, 10:-10]
                loss += F.mse_loss(x, y)
        loss = loss / 16
        return loss
