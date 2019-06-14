import torch
import torch.nn as nn


def mse_psnr(mse_loss, r=255.):
    return 10 * torch.log10(r ** 2 / mse_loss)


class PSNR(nn.Module):
    def __init__(self, r=255.):
        super().__init__()
        self.r = r
        self.mse = nn.MSELoss()
        self.require_grad = False

    def forward(self, hr, sr, trim=10, round_clip=False):
        diff = (sr - hr) / self.r
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)
        shave = 10 #scale + 6

        valid = diff[..., shave:-shave, shave:-shave]
        mse = valid.pow(2).mean()
        return -10 * torch.log10(mse)
        '''
        if round_clip:
            hr = torch.clamp(torch.round(hr), 0., 255.) / 255.
            hr = hr[:, :, trim:-trim, trim:-trim]
            hr = torch.round(hr[:, 0, :, :] * 65.481 + hr[:, 1, :, :] * 128.553 + hr[:, 2, :, :] * 24.966 + 16)
            sr = torch.clamp(torch.round(sr), 0., 255.) / 255.
            sr = sr[:, :, trim:-trim, trim:-trim]
            sr = torch.round(sr[:, 0, :, :] * 65.481 + sr[:, 1, :, :] * 128.553 + sr[:, 2, :, :] * 24.966 + 16)
        return mse_psnr(self.mse(hr, sr), r=self.r)
    
    def forward(self, hr, sr, round_clip=True):
        if round_clip:
            hr = torch.clamp(torch.round(hr), 0., 255.)
            hr = torch.round(hr[:, 0, :, :] * 0.299 + hr[:, 1, :, :] * 0.587 + hr[:, 2, :, :] * 0.114)
            sr = torch.clamp(torch.round(sr), 0., 255.)
            sr = torch.round(sr[:, 0, :, :] * 0.299 + sr[:, 1, :, :] * 0.587 + sr[:, 2, :, :] * 0.114)
        return mse_psnr(self.mse(hr, sr), r=255)
    '''
