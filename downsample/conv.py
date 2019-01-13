import torch.nn as nn


class ConvolutionDownscale(nn.Module):
    def __init__(self, scale=4, channels=3):
        super().__init__()
        filter_height, filter_width = scale * 4, scale * 4
        stride = scale
        pad_along_height = max(filter_height - stride, 0)
        pad_along_width = max(filter_width - stride, 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        
        self.model = nn.Sequential(
            nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom)),
            nn.Conv2d(in_channels=channels,
                      out_channels=channels * scale ** 2,
                      kernel_size=scale * 4,
                      stride=scale,
                      padding=0, groups=channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels * scale ** 2,
                      out_channels=channels,
                      kernel_size=1,
                      stride=1,
                      padding=0, groups=channels)
        )

    def forward(self, x):
        return self.model(x)
