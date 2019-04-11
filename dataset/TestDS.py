from torch.utils.data import Dataset
import os
from imageio import imread
import numpy as np
from skimage.color import gray2rgb


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
