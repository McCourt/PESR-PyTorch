import torch
from torch import nn
from time import time
import json


def since(begin):
    return time() - begin


class Timer(object):
    def __init__(self):
        self.begin = time()

    def report(self):
        return since(self.begin)

    def refresh(self):
        self.begin = time()


def load_parameters(path):
    with open(path, 'r') as f:
        return json.load(f)

    
def fourier_transform(img):
    img = torch.squeeze(img[:, 0, :, :] * 65.481 + img[:, 1, :, :] * 128.553 + img[:, 2, :, :] * 24.966 + 16)
    return torch.rfft(img, 2)