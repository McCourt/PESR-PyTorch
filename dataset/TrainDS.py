from torch.utils.data import Dataset
import os
from imageio import imread
import numpy as np
from skimage.color import gray2rgb
import pandas as pd


class SRTrainDataset(Dataset):
    def __init__(self, scale, dataset=None, name_dict='dataset/name_dic.csv', h=40, w=40, num_per=600, img_format='png',
                 rgb_shuffle=True, rotate=True, flip=True, shuffle_rate=0.1, rotate_rate=0.1, flip_rate=0.1):
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
        df = pd.read_csv(name_dict)
        # df = df[df.usage == 'train']
        df = df[df.scale == scale]
        if dataset is not None:
            if type(dataset) is str:
                df = df[df.dataset == dataset]
            else:
                df = df[df.dataset.isin(dataset)]
        self.df = df.sort_values('name').reset_index()

        self.num_img = len(self.df.index)
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
            self.name = self.df.name[self.id]

            hr = np.array(imread(self.df.HR[self.id], as_gray=False)).astype(np.float32)
            lr = np.array(imread(self.df.LR[self.id], as_gray=False)).astype(np.float32)

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
            lr = np.rot90(lr, angle).copy()
            hr = np.rot90(hr, angle).copy()
        elif self.flip and np.random.rand() < self.flip_rate:
            if np.random.rand() < .5:
                lr = np.flipud(lr).copy()
                hr = np.flipud(hr).copy()
                # lr = lr[::-1, :, :]
                # hr = hr[::-1, :, :]
            else:
                lr = np.fliplr(lr).copy()
                hr = np.fliplr(hr).copy()
                # lr = lr[:, ::-1, :]
                # hr = hr[:, ::-1, :]
        elif self.rgb_shuffle and np.random.rand() < self.shuffle_rate:
            new_order = np.random.permutation([0, 1, 2])
            lr = lr[:, :, new_order].copy()
            hr = hr[:, :, new_order].copy()
        else:
            pass
        return {'hr': np.moveaxis(hr, -1, 0), 'lr': np.moveaxis(lr, -1, 0), 'name': self.name}
