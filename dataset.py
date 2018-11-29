from torch.utils.data import Dataset, DataLoader
import sys, os
from imageio import imread
import numpy as np
from scipy.misc import imresize

class ResolutionDataset(Dataset):
    def __init__(self, hr_dir, lr_dir,
                h=40, w=40, scale=4, format='png', nwhc=True, num_per=400,
                upsample=False, rgb_shuffle=False, rotation=False, flip=False):
        self.hr_names = sorted([os.path.join(hr_dir, i) for i in os.listdir(hr_dir) if format in i])
        self.lr_names = sorted([os.path.join(lr_dir, i) for i in os.listdir(lr_dir) if format in i])
        # assert len(self.hr_names) == len(self.lr_names)
        self.nwhc = nwhc
        self.num_per = num_per
        self.id, self.hr, self.lr, self.name = None, None, None, None
        self.h, self.w = h, w
        self.upsample, self.rgb_shuffle, self.rotation, self.flip = upsample, rgb_shuffle, rotation, flip
        self.scale, self.lrc, self.lrh, self.lrw = scale, None, None, None
        self.begin = True

    def __len__(self):
        return (len(self.hr_names) * self.num_per)

    def __getitem__(self, idx):
        if self.begin or idx % self.num_per == 0:
            self.begin = False
            self.id = idx // self.num_per
            assert self.hr_names[self.id].split('/')[-1] == self.lr_names[self.id].split('/')[-1].replace('x4', '')
            self.name = self.hr_names[self.id].split('/')[-1]
            self.hr = imread(self.hr_names[self.id])
            self.lr = imread(self.lr_names[self.id])
            if self.upsample:
                self.lr = imresize(np.array(self.lr), size=self.hr.shape, interp='bicubic')
            if self.nwhc:
                self.hr = np.moveaxis(self.hr, -1, 0)
                self.lr = np.moveaxis(self.lr, -1, 0)
            self.lrc, self.lrh, self.lrw = self.lr.shape
        rand_h, rand_w = np.random.randint(0, self.lrh - self.h), np.random.randint(0, self.lrw - self.w)
        lr = self.lr[:, rand_h:rand_h + self.h, rand_w:rand_w + self.w]
        if not self.upsample:
            hr = self.hr[:, rand_h * self.scale:(rand_h + self.h) * self.scale, rand_w * self.scale:(rand_w + self.w) * self.scale]
        else:
            hr = self.hr[:, rand_h:rand_h + self.h, rand_w:rand_w + self.w]
        return {'hr': np.array(hr).astype(np.float32), 'lr': np.array(lr).astype(np.float32)}
