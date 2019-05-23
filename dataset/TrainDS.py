from torch.utils.data import Dataset
import os
from imageio import imread
import numpy as np
from skimage.color import gray2rgb


class SRTrainDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, h=40, w=40, scale=1, num_per=600,
                 img_format='png', hr_formatter=None, lr_formatter=None,
                 rgb_shuffle=False, rotate=False, flip=False,
                 shuffle_rate=0.1, rotate_rate=0.1, flip_rate=0.1):
        """
        This method is implemented to deal with data inputs in the SR training process.
        :param hr_dir: directory name for HR images
        :param lr_dir: directory name for LR images
        :param h: window height for random cropping in LR image, corresponding HR window height is h * scale
        :param w: window width for random cropping in LR image, corresponding HR window is width w * scale
        :param scale: scale ratio between HR and LR, expect to be greater than 1. Default: 1 for residual net
        :param num_per: number of patches sampled from each image in data directory
        :param img_format: type of format of images. Default: PNG
        :param hr_formatter: name formatter for HR images
        :param lr_formatter: name formatter for LR images
        :param rgb_shuffle: if to random shuffle RGB channels at Bernoulli probability of shuffle_rate. Default: False
        :param rotate: if to random shuffle image at Bernoulli probability of rotate_rate. Default: False
                        1/3 probability for 90, 180 or 270 degree of rotation
        :param flip: if to random flip image at Bernoulli probability of flip_rate. Default: False
                        1/2 probability for up and down flip or left and right flip
        :param shuffle_rate: probability to shuffle RGB channels
        :param rotate_rate: probability to rotate images
        :param flip_rate: probability to flip images
        """
        self.hr_formatter = lambda x: x if hr_formatter is None else hr_formatter
        self.lr_formatter = lambda x: x.replace('x4', '') if lr_formatter is None else lr_formatter

        hr_names = sorted([i for i in os.listdir(hr_dir) if img_format in i])
        lr_names = sorted([i for i in os.listdir(lr_dir) if img_format in i])
        assert len(hr_names) == len(lr_names)
        assert all(self.hr_formatter(i) == self.lr_formatter(j) for i, j in zip(hr_names, lr_names))
        self.img_names = hr_names

        self.hr_names = [os.path.join(hr_dir, hr_name) for hr_name in hr_names]
        self.lr_names = [os.path.join(lr_dir, lr_name) for lr_name in lr_names]
        self.num_img = len(self.img_names)

        self.scale, self.num_per = scale, num_per
        self.h, self.w = h, w
        self.rgb_shuffle, self.rotate, self.flip = rgb_shuffle, rotate, flip
        self.shuffle_rate, self.rotate_rate, self.flip_rate = shuffle_rate, rotate_rate, flip_rate

        self.id = None
        self.hr, self.lr = None, None
        self.lrh, self.lrw, _ = None, None, None
        self.hrh, self.hrw, _ = None, None, None

    def __len__(self):
        return self.num_img * self.num_per

    def __getitem__(self, idx):
        """
        Give one image patch in HR/LR/name pair.
        :param idx: index of image patch pairs, which should be less than size of dataset
        :return: dictionary with keys of "hr", "lr" and "name"
        """
        if self.id is None or idx // self.num_per != self.id:
            self.id = idx // self.num_per
            self.name = self.img_names[self.id]

            hr = np.array(imread(self.hr_names[self.id], as_gray=False)).astype(np.float32)
            lr = np.array(imread(self.lr_names[self.id], as_gray=False)).astype(np.float32)

            if len(hr.shape) == 2:
                hr = gray2rgb(hr)
            if len(lr.shape) == 2:
                lr = gray2rgb(lr)

            self.hr, self.lr = hr, lr
            self.lrh, self.lrw, _ = self.lr.shape
            self.hrh, self.hrw, _ = self.hr.shape

        lr_x = np.random.randint(0, self.lrh - self.h)
        lr_y = np.random.randint(0, self.lrw - self.w)
        hr_x = lr_x * self.scale
        hr_y = lr_y * self.scale
        lr = self.lr[lr_x:lr_x + self.h, lr_y:lr_y + self.w, :]
        hr = self.hr[hr_x:hr_x + self.h * self.scale, hr_y:hr_y + self.w * self.scale, :]

        if self.rotate and np.random.rand() < self.rotate_rate:
            angle = np.random.choice([1, 2, 3])
            lr = np.rot90(lr, angle)
            hr = np.rot90(hr, angle)
        elif self.flip and np.random.rand() < self.flip_rate:
            if np.random.rand() < .5:
                lr = np.flipud(lr)
                hr = np.flipud(hr)
            else:
                lr = np.fliplr(lr)
                hr = np.fliplr(hr)
        elif self.rgb_shuffle and np.random.rand() < self.shuffle_rate:
            new_order = np.random.permutation([0, 1, 2])
            lr = lr[:, :, new_order]
            hr = hr[:, :, new_order]
        else:
            pass
        return {'hr': np.moveaxis(hr, -1, 0), 'lr': np.moveaxis(lr, -1, 0), 'name': self.name}
