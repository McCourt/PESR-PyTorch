import torch.nn as nn
from importlib import import_module
import json


class Model(nn.Module):
    def __init__(self, name, mode, **kwargs):
        super().__init__()
        with open('model/models.json', 'r') as f:
            params = json.load(f)
        print(params)

        if mode not in params.keys():
            raise ValueError('Wrong mode. Try {}'.format(', '.join(params.keys())))
        elif name not in params[mode]:
            raise ValueError('Wrong model name. Try {}'.format(', '.join(params[mode].keys())))
        path = '.'.join(['model', mode, params[mode][name.lower()]])
        module = getattr(import_module(path), 'Model')
        self.model = module(**kwargs)

    def forward(self, x):
        return self.model(x)
