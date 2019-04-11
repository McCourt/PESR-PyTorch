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


def load_parameters(path='./parameter.json'):
    with open(path, 'r') as f:
        return json.load(f)
