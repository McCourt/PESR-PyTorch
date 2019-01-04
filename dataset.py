from torch.utils.data import Dataset, DataLoader
import sys, os
from imageio import imread
import numpy as np
from scipy.misc import imresize

default_parse = lambda x: x


class ResolutionDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, h=40, w=40, scale=4, img_format='png', num_per=400,
                 hr_parse=default_parse, lr_parse=default_parse,
                 rgb_shuffle=False, rotation=False, flip=False):
        hr_names = sorted([os.path.join(hr_dir, hr_parse(i)) for i in os.listdir(hr_dir) if img_format in i])
        lr_names = sorted([os.path.join(lr_dir, lr_parse(i)) for i in os.listdir(lr_dir) if img_format in i])
        assert len(hr_names) == len(lr_names)
        assert all(i == j for i, j in zip(hr_names, lr_names))
        self.hr_names = [os.path.join(hr_dir, i) for i in hr_names]
        self.lr_names = [os.path.join(lr_dir, i) for i in lr_names]

        if rotation:
            num_per += 3
        if flip:
            num_per += 1
        if rgb_shuffle:
            num_per += 6
        self.scale, self.num_per = scale, num_per
        self.h, self.w = h, w
        self.rgb_shuffle, self.rotation, self.flip = rgb_shuffle, rotation, flip

        self.id, self.hr, self.lr, self.name = None, None, None, None
        self.lrc, self.lrh, self.lrw = None, None, None

    def __len__(self):
        return len(self.hr_names) * self.num_per

    def __getitem__(self, idx):
        if idx % self.num_per == 0:
            self.id = idx // self.num_per
            self.name = self.hr_names[self.id].split('/')[-1]
            self.hr = np.moveaxis(imread(self.hr_names[self.id]), -1, 0)
            self.lr = np.moveaxis(imread(self.lr_names[self.id]), -1, 0)
            self.lrc, self.lrh, self.lrw = self.lr.shape
        rand_h, rand_w = np.random.randint(0, self.lrh - self.h), np.random.randint(0, self.lrw - self.w)
        lr = self.lr[:, rand_h:rand_h + self.h, rand_w:rand_w + self.w]
        hr = self.hr[:, rand_h * self.scale:(rand_h + self.h) * self.scale,
             rand_w * self.scale:(rand_w + self.w) * self.scale]
        return {'hr': np.array(hr).astype(np.float32), 'lr': np.array(lr).astype(np.float32)}


class ImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, img_format='png', hr_parse=default_parse, lr_parse=default_parse):
        hr_names = sorted([os.path.join(hr_dir, i) for i in os.listdir(hr_dir) if img_format in i])
        lr_names = sorted([os.path.join(lr_dir, i) for i in os.listdir(lr_dir) if img_format in i])
        assert len(hr_names) == len(lr_names)
        assert all(hr_parse(i.split('/')[-1]) == lr_parse(j.split('/')[-1]) for i, j in zip(hr_names, lr_names))
        self.hr_names = [os.path.join(hr_dir, i) for i in hr_names]
        self.lr_names = [os.path.join(lr_dir, i) for i in lr_names]

    def __len__(self):
        return len(self.hr_names)

    def __getitem__(self, item):
        hr = np.moveaxis(imread(self.hr_names[item]), -1, 0)
        lr = np.moveaxis(imread(self.lr_names[item]), -1, 0)
        return {'hr': np.array(hr).astype(np.float32), 'lr': np.array(lr).astype(np.float32)}
