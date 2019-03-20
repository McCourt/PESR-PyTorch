import torch
import torch.nn.functional as F


class DownSampler(torch.nn.Module):
    def __init__(self, device=torch.device('cuda'), num_channels=20, activation=F.relu):
        super(DownSampler, self).__init__()
        self.activation = activation
        self.num_channels = num_channels
        # Reuse filters because each channel should be using the same set of filters
        self.weight44_1 = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.zeros(num_channels, 1, 4, 4)).to(device))
        self.weight44_2 = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.zeros(num_channels, num_channels, 4, 4)).to(device))
        self.weight_out = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.zeros(1, num_channels, 3, 3)).to(device))

    def forward(self, x):
        pad_1 = F.pad(x, (1, 1, 1, 1), mode='reflect')
        channels = torch.split(pad_1, 1, dim=1)  # split by channel RGB
        out_1 = [self.activation(F.conv2d(l, self.weight44_1, stride=2)) for l in channels]

        pad_2 = [F.pad(l, (1, 1, 1, 1), mode='reflect') for l in out_1]
        out_2 = [self.activation(F.conv2d(l, self.weight44_2, stride=2)) for l in pad_2]

        out_3 = [self.activation(F.conv2d(l, self.weight_out, padding=1)) for l in out_2]

        out = torch.cat(out_3, dim=1)

        return out