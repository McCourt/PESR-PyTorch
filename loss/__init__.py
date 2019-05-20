from loss.laploss import LapLoss
from loss.dsloss import DownScaleLoss
from loss.discloss import GanLoss
from loss.tdsloss import TrainedDownScaleLoss
from loss.shiftloss import ShiftLoss, TrainedShiftLoss
from loss.regloss import RegularizationLoss
from loss.psnr import PSNR, mse_psnr
from torch import nn


class Loss(nn.Module):
    def __init__(self, loss_fn='1.0*l1+1.0*decay_ds'):
        super().__init__()

    def forward(self, hr, lr, sr):
        pass