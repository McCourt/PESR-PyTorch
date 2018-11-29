import torch
from torch import nn
from torch.nn import functional as F

class BicubicDownSample(nn.Module):
    def bicubic_kernel(self, x, a=-0.5):
        """
        This equation is exactly copied from the website below:
        https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic
        """
        abs_x = torch.abs(x)
        if abs_x <= 1:
            return (a + 2) * torch.pow(abs_x, 3) - (a + 3) * torch.pow(abs_x, 2) + 1
        elif 1 < abs_x < 2:
            return a * torch.pow(abs_x, 3) - 5 * a * torch.pow(abs_x, 2) + 8 * a * abs_x - 4 * a
        else:
            return 0.0

    def __init__(self, factor=4):
        super().__init__()
        self.factor = factor
        size = factor * 4
        k = torch.tensor([self.bicubic_kernel((i - torch.floor(torch.tensor(size / 2)) + 0.5) / factor) for i in range(size)],dtype=torch.float32)
        k = k / torch.sum(k)
        k = torch.einsum('i,j->ij', (k, k))
        k = torch.reshape(k, shape=(1, 1, size, size))
        self.k = torch.cat([k, k, k], dim=0)

    def forward(self, x, nhwc = False):
        filter_height = self.factor * 4
        filter_width = self.factor * 4
        stride = self.factor

        pad_along_height = max(filter_height - stride, 0)
        pad_along_width = max(filter_width - stride, 0)
        filters = self.k

        # compute actual padding values for each side
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        # apply mirror padding
        if nhwc:
            x = torch.transpose(torch.transpose(x, 2, 3), 1, 2)   # NHWC to NCHW
        pad = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        x = pad(x)

        # downsampling performed by strided convolution
        x = F.conv2d(input=x, weight=filters.type('torch.cuda.DoubleTensor').cuda(), stride=stride, groups=3)
        if nhwc:
            return torch.transpose(torch.transpose(x, 1, 3), 1, 2)
        else:
            return x
