import torch
import torch.nn as nn
import torch.nn.functional as F
from model.downscaler.bicubic import BicubicDownSample


class ShiftLoss(nn.Module):
    def __init__(self, kernel_dir='checkpoints/shifts.ckpt'):
        super(ShiftLoss, self).__init__()
        self.kernel_dict = torch.load(kernel_dir)
        self.bicubic = BicubicDownSample()

    def forward(self, hr, lr):
        loss = 0
        for i in range(-2, 2):
            for j in range(-2, 2):
                kernel = self.kernel_dict[(i, j)].type('torch.cuda.FloatTensor')
                x = F.conv2d(input=lr, weight=kernel, stride=1)
                y = self.bicubic(hr.roll((-i, -j), (2, 3)))[:, :, 10:-10, 10:-10]
                loss += F.mse_loss(x, y)
        loss = loss / 16
        return loss


class TTOShift(torch.nn.Module):
    def __init__(self, num_channels=(64, 128, 256, 512, 48), activation=F.relu):
        super(TTOShift, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(3, num_channels[0], (1, 1), stride=1, padding=0)
        self.conv2 = nn.Conv2d(num_channels[0], num_channels[1], (3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels[1], num_channels[2], (5, 5), stride=1, padding=2)
        self.conv4 = nn.Conv2d(num_channels[2], num_channels[3], (3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_channels[3], num_channels[4], (5, 5), stride=1, padding=2)

    def forward(self, x):
        out1 = self.activation(self.conv1(x))
        out2 = self.activation(self.conv2(out1))
        out3 = self.activation(self.conv3(out2))
        out4 = self.activation(self.conv4(out3))
        out5 = self.activation(self.conv5(out4))

        return out5


class TrainedShiftLoss(nn.Module):
    def __init__(self, checkpoint='checkpoints/TTOShift.pth'):
        super(TrainedShiftLoss, self).__init__()
        self.shifts = TTOShift()
        self.shifts.load_state_dict(torch.load(checkpoint))
        self.bicubic = BicubicDownSample()

    def forward(self, hr, lr):
        loss = 0
        hr_down = []
        for i in range(-2, 2):
            for j in range(-2, 2):
                y = self.bicubic(hr.roll((i, j), (2, 3)))
                hr_down.append(y)
        hr_down_ = torch.cat(hr_down, dim=1)
        shifted = self.shifts(lr)
        loss = F.mse_loss(shifted, hr_down_)
        return loss
