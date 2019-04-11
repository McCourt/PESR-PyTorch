import torch
import torch.nn as nn
from importlib import import_module
import json


def load_checkpoint(load_dir, map_location=None):
    try:
        print('loading checkpoint from {}'.format(load_dir))
        checkpoint = torch.load(load_dir, map_location=map_location)
        print('loading successful')
        return checkpoint
    except:
        print('No checkpoint and begin new training')


class Model(nn.Module):
    def __init__(self, name, mode, checkpoint, train=True, map_location=None, **kwargs):
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
        print('Number of parameters of SR model: {:.2E}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        try:
            sr_checkpoint = load_checkpoint(load_dir=checkpoint, map_location=map_location)
            if sr_checkpoint is None:
                print('Start new training for {} model'.format(mode))
            else:
                print('{} model recovering from checkpoints'.format(mode))
                self.model.load_state_dict(sr_checkpoint['model'])
        except:
            raise ValueError('Checkpoint not found.')

        for param in self.model.parameters():
            param.requires_grad = False if not train else param.requires_grad

    def forward(self, x):
        return self.model(x)
