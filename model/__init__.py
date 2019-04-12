import torch
import torch.nn as nn
from importlib import import_module
from src.helper import load_parameters
import os


def save_checkpoint(state_dict, save_dir):
    try:
        torch.save(state_dict, save_dir)
    except:
        raise Exception('checkpoint saving failure')


def load_checkpoint(load_dir, map_location=None):
    try:
        print('loading checkpoint from {}'.format(load_dir))
        checkpoint = torch.load(load_dir, map_location=map_location)
        print('loading successful')
        return checkpoint
    except:
        print('No checkpoint and begin new training')


def report_num_params(model):
    print('Number of parameters of model: {:.2E}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))


class Model(nn.Module):
    def __init__(self, name, mode, checkpoint=None, train=True, map_location=None, **kwargs):
        super().__init__()
        params = load_parameters(path='model/models.json')

        if mode not in params.keys():
            raise ValueError('Wrong mode. Try {}'.format(', '.join(params.keys())))
        elif name not in params[mode]:
            raise ValueError('Wrong model name. Try {}'.format(', '.join(params[mode].keys())))
        path = '.'.join(['model', mode])
        module = getattr(import_module(path), params[mode][name.lower()])
        self.model = module(**kwargs)
        self.model = nn.DataParallel(self.model).cuda()
        report_num_params(self.model)

        self.checkpoint, self.mode, self.map_location = checkpoint, mode, map_location
        self.load_checkpoint()

        self.is_train = train
        if not self.is_train:
            print('Disabling auto gradient and switching to TEST mode')
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print('{} model is ready for training'.format(mode))

    def load_checkpoint(self):
        if self.checkpoint is not None and os.path.isfile(self.checkpoint):
            try:
                print('loading checkpoint from {}'.format(self.checkpoint))
                ckpt = torch.load(self.checkpoint, map_location=self.map_location)
                if ckpt is None:
                    print('No checkpoint and start new training for {} model'.format(self.mode))
                else:
                    print('loading successful and recovering checkpoints for {} model'.format(self.mode))
                    self.load_state_dict(ckpt)
                    print('Checkpoint loaded successfully')
            except:
                raise ValueError('Wrong Checkpoint path or loaded erroneously')
        else:
            print('No checkpoint and start new training for {} model'.format(self.mode))

    def save_checkpoint(self, add_time=False):
        try:
            torch.save(self.state_dict(), self.checkpoint)
        except:
            raise Exception('checkpoint saving failed')

    def forward(self, x):
        output = self.model(x)
        if not self.is_train:
            output = torch.clamp(torch.round(output), 0., 255.)
        return output
