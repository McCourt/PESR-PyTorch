import torch
import torch.nn as nn
from importlib import import_module
import json


def load_checkpoint(load_dir, map_location=None):
    try:
        print('loading checkpoint from {}'.format(load_dir))
        checkpoint = torch.load(load_dir, map_location=map_location)
        return checkpoint
    except:
        print('begin new training')


def report_num_params(model):
    print('Number of parameters of model: {:.2E}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))


class Model(nn.Module):
    def __init__(self, name, mode, checkpoint, train=True, map_location=None, **kwargs):
        super().__init__()
        with open('model/models.json', 'r') as f:
            params = json.load(f)

        if mode not in params.keys():
            raise ValueError('Wrong mode. Try {}'.format(', '.join(params.keys())))
        elif name not in params[mode]:
            raise ValueError('Wrong model name. Try {}'.format(', '.join(params[mode].keys())))
        path = '.'.join(['model', mode, params[mode][name.lower()][0]])
        module = getattr(import_module(path), params[mode][name.lower()][1])
        self.model = module(**kwargs)
        self.model = nn.DataParallel(self.model).cuda()
        report_num_params(self.model)

        try:
            print('loading checkpoint from {}'.format(checkpoint))
            ckpt = torch.load(checkpoint, map_location=map_location)
            if ckpt is None:
                print('No checkpoint and start new training for {} model'.format(mode))
            else:
                print('loading successful and recovering checkpoints for {} model'.format(mode))
                self.model.load_state_dict(ckpt['model'])
                print('Checkpoint loaded successfully')
        except:
            raise ValueError('Wrong Checkpoint path or loaded erroneously')

        self.is_train = train
        if not self.is_train:
            print('Disabling auto gradient and switching to TEST mode')
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print('{} model is ready for training'.format(mode))

    def forward(self, x):
        output = self.model(x)
        if not self.is_train:
            output = torch.clamp(torch.round(output), 0., 255.)
        return output
