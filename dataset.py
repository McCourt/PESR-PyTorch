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
        self.lr_formatter = lambda x: x if lr_formatter is None else lr_formatter

        hr_names = sorted([self.hr_formatter(i) for i in os.listdir(hr_dir) if img_format in i])
        lr_names = sorted([self.lr_formatter(i) for i in os.listdir(lr_dir) if img_format in i])
        assert len(hr_names) == len(lr_names)
        assert all(i == j for i, j in zip(hr_names, lr_names))
        self.img_names = hr_names

        self.hr_names = [os.path.join(hr_dir, hr_name) for hr_name in hr_names]
        self.lr_names = [os.path.join(lr_dir, lr_name) for lr_name in lr_names]
        self.num_img = len(self.img_names)

        self.scale, self.num_per = scale, num_per
        self.h, self.w = h, w
        self.rgb_shuffle, self.rotate, self.flip = rgb_shuffle, rotate, flip
        self.shuffle_rate, self.rotate_rate, self.flip_rate = shuffle_rate, rotate_rate, flip_rate

        self.id = 0
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

    def __len__(self):
        return self.num_img * self.num_per

    def __getitem__(self, idx):
        """
        Give one image patch in HR/LR/name pair.
        :param idx: index of image patch pairs, which should be less than size of dataset
        :return: dictionary with keys of "hr", "lr" and "name"
        """
        if idx % self.num_per == 0 and self.id != 0:
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
        print(self.lrh)
        lr_h_center = np.random.randint(self.h // 2, self.lrh - self.h // 2)
        lr_w_center = np.random.randint(self.w // 2, self.lrw - self.w // 2)
        hr_h_center = (lr_h_center - self.lrh // 2) * self.scale + self.hrh // 2
        hr_w_center = (lr_w_center - self.lrw // 2) * self.scale + self.hrw // 2
        lr_h_from, lr_w_from = lr_h_center - self.h // 2, lr_w_center - self.w // 2
        hr_h_from, hr_w_from = hr_h_center - self.h // 2 * self.scale, hr_w_center - self.w // 2 * self.scale
        lr_h_to, lr_w_to = lr_h_from + self.h, lr_w_from + self.w
        hr_h_to, hr_w_to = hr_h_from + self.h * self.scale, hr_w_from + self.w * self.scale

        lr = self.lr[lr_h_from:lr_h_to, lr_w_from:lr_w_to, :]
        hr = self.hr[hr_h_from:hr_h_to, hr_w_from:hr_w_to, :]

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


class SRTestDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, img_format='png', hr_formatter=None, lr_formatter=None):
        """
        This method is implemented to deal with data inputs in the SR validation or testing process.
        :param hr_dir: directory name for HR images
        :param lr_dir: directory name for LR images
        :param img_format: type of format of images. Default: PNG
        :param hr_formatter: name formatter for HR images
        :param lr_formatter: name formatter for LR images
        """
        self.hr_formatter = lambda x: x if hr_formatter is None else hr_formatter
        self.lr_formatter = lambda x: x if lr_formatter is None else lr_formatter

        hr_names = sorted([self.hr_formatter(i) for i in os.listdir(hr_dir) if img_format in i])
        lr_names = sorted([self.lr_formatter(i) for i in os.listdir(lr_dir) if img_format in i])
        assert len(hr_names) == len(lr_names)
        assert all(i == j for i, j in zip(hr_names, lr_names))
        self.img_names = hr_names

        self.hr_names = [os.path.join(hr_dir, hr_name) for hr_name in hr_names]
        self.lr_names = [os.path.join(lr_dir, lr_name) for lr_name in lr_names]
        self.num_img = len(self.hr_names)

    def __len__(self):
        return self.num_img

    def __getitem__(self, idx):
        """
        Give whole images in HR/LR/name pair.
        :param idx: index of image pairs, which should be less than size of dataset
        :return: dictionary with keys of "hr", "lr" and "name"
        """
        hr = np.array(imread(self.hr_names[idx], as_gray=False)).astype(np.float32)
        lr = np.array(imread(self.lr_names[idx], as_gray=False)).astype(np.float32)

        if len(hr.shape) == 2:
            hr = gray2rgb(hr)
        if len(lr.shape) == 2:
            lr = gray2rgb(lr)

        return {'hr': np.moveaxis(hr, -1, 0), 'lr': np.moveaxis(lr, -1, 0), 'name': self.img_names[idx]}


class SRTTODataset(Dataset):
    def __init__(self, hr_dir, lr_dir, sr_dir, img_format='png', hr_formatter=None, lr_formatter=None, sr_formatter=None):
        """
        This method is implemented to deal with data inputs in the test time optimization.
        :param hr_dir: directory name for HR images
        :param lr_dir: directory name for LR images
        :param sr_dir: directory name for SR images
        :param img_format: type of format of images. Default: PNG
        :param hr_formatter: name formatter for HR images
        :param lr_formatter: name formatter for LR images
        :param sr_formatter: name formatter for SR images
        """
        self.hr_formatter = lambda x: x if hr_formatter is None else hr_formatter
        self.lr_formatter = lambda x: x if lr_formatter is None else lr_formatter
        self.sr_formatter = lambda x: x if sr_formatter is None else sr_formatter

        hr_names = sorted([self.hr_formatter(i) for i in os.listdir(hr_dir) if img_format in i])
        lr_names = sorted([self.lr_formatter(i) for i in os.listdir(lr_dir) if img_format in i])
        sr_names = sorted([self.sr_formatter(i) for i in os.listdir(sr_dir) if img_format in i])
        assert len(hr_names) == len(lr_names) == len(sr_names)
        assert all(i == j == k for i, j, k in zip(hr_names, lr_names, sr_names))
        self.img_names = hr_names

        self.hr_names = [os.path.join(hr_dir, hr_name) for hr_name in hr_names]
        self.lr_names = [os.path.join(lr_dir, lr_name) for lr_name in lr_names]
        self.sr_names = [os.path.join(sr_dir, sr_name) for sr_name in sr_names]
        self.num_img = len(self.hr_names)

    def __len__(self):
        return self.num_img

    def __getitem__(self, idx):
        """
        Give whole images in HR/LR/SR/name pair.
        :param idx: index of image pairs, which should be less than size of dataset
        :return: dictionary with keys of "hr", "lr", "sr" and "name"
        """
        hr = np.array(imread(self.hr_names[idx], as_gray=False)).astype(np.float32)
        lr = np.array(imread(self.lr_names[idx], as_gray=False)).astype(np.float32)
        sr = np.array(imread(self.sr_names[idx], as_gray=False)).astype(np.float32)

        if len(hr.shape) == 2:
            hr = gray2rgb(hr)
        if len(lr.shape) == 2:
            lr = gray2rgb(lr)
        if len(sr.shape) == 2:
            sr = gray2rgb(sr)

        return {'hr': np.moveaxis(hr, -1, 0), 'lr': np.moveaxis(lr, -1, 0),
                'sr': np.moveaxis(sr, -1, 0), 'name': self.img_names[idx]}
