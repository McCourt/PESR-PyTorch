from torch.utils.data import Dataset
import os
from imageio import imread
import numpy as np
from skimage.color import gray2rgb


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
