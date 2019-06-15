from torch.utils.data import Dataset
import os
from imageio import imread
import numpy as np
from skimage.color import gray2rgb
import pandas as pd


class SRTestDataset(Dataset):
    def __init__(self, dataset, scale, name_dict='dataset/name_dic.csv', img_format='png', hr_formatter=None, lr_formatter=None):
        """
        This method is implemented to deal with data inputs in the SR validation or testing process.
        :param hr_dir: directory name for HR images
        :param lr_dir: directory name for LR images
        :param img_format: type of format of images. Default: PNG
        :param hr_formatter: name formatter for HR images
        :param lr_formatter: name formatter for LR images
        """
        # self.hr_formatter = lambda x: x if hr_formatter is None else hr_formatter
        # # self.lr_formatter = lambda x: x if lr_formatter is None else lr_formatter
        # self.lr_formatter = lambda x: x.replace('x4', '') if lr_formatter is None else lr_formatter
        # hr_names = sorted([i for i in os.listdir(hr_dir) if img_format in i])
        # lr_names = sorted([i for i in os.listdir(lr_dir) if img_format in i])
        # assert len(hr_names) == len(lr_names)
        # assert all(self.hr_formatter(i) == self.lr_formatter(j) for i, j in zip(hr_names, lr_names))
        # self.img_names = hr_names
        #
        # self.hr_names = [os.path.join(hr_dir, hr_name) for hr_name in hr_names]
        # self.lr_names = [os.path.join(lr_dir, lr_name) for lr_name in lr_names]
        # self.num_img = len(self.hr_names)
        df = pd.read_csv(name_dict)
        self.df = df[(df.usage == 'valid') & (df.scale == scale) & (df.dataset == dataset)].reset_index()

        # hr_names = sorted([i for i in os.listdir(hr_dir) if img_format in i])
        # lr_names = sorted([i for i in os.listdir(lr_dir) if img_format in i])
        # assert len(hr_names) == len(lr_names)
        # assert all(self.hr_formatter(i) == self.lr_formatter(j) for i, j in zip(hr_names, lr_names))
        # self.img_names = hr_names
        #
        # self.hr_names = [os.path.join(hr_dir, hr_name) for hr_name in hr_names]
        # self.lr_names = [os.path.join(lr_dir, lr_name) for lr_name in lr_names]

        self.num_img = len(self.df.index)

    def __len__(self):
        return self.num_img

    def __getitem__(self, idx):
        """
        Give whole images in HR/LR/name pair.
        :param idx: index of image pairs, which should be less than size of dataset
        :return: dictionary with keys of "hr", "lr" and "name"
        """
        hr = np.array(imread(self.df.HR[idx], as_gray=False)).astype(np.float32)
        lr = np.array(imread(self.df.LR[idx], as_gray=False)).astype(np.float32)

        if len(hr.shape) == 2:
            hr = gray2rgb(hr)
        if len(lr.shape) == 2:
            lr = gray2rgb(lr)

        return {'hr': np.moveaxis(hr, -1, 0), 'lr': np.moveaxis(lr, -1, 0), 'name': self.df.name[idx]}
