from torch.utils.data import Dataset
import os
from imageio import imread
import numpy as np
from skimage.color import gray2rgb
import pandas as pd


class SRTestDataset(Dataset):
    def __init__(self, scale, dataset=None, name_dict='dataset/name_dic.csv', img_format='png', hr_formatter=None, lr_formatter=None):
        """
        This method is implemented to deal with data inputs in the SR validation or testing process.
        :param hr_dir: directory name for HR images
        :param lr_dir: directory name for LR images
        :param img_format: type of format of images. Default: PNG
        :param hr_formatter: name formatter for HR images
        :param lr_formatter: name formatter for LR images
        """
        df = pd.read_csv(name_dict)
        df = df[(df.usage == 'valid') & (df.scale == scale)]
        if dataset is not None:
            df = df[df.dataset == dataset]
        self.df = df.sort_values('name').reset_index()
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
