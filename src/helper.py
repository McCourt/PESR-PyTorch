import torch
from torch import nn
from time import time
import json
from datetime import datetime


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


def report_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def output_report(report, end='\n'):
    print('{}: {}'.format(report_time(), report), end=end)
