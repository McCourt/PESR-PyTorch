import torch
import torch.nn as nn
from importlib import import_module
from src.helper import load_parameters, Timer
from loss.psnr import PSNR
import os
import numpy as np
from dataset import SRTrainDataset, SRTestDataset
from torch.utils.data import DataLoader
from imageio import imwrite


def save_checkpoint(state_dict, save_dir):
    try:
        torch.save(state_dict, save_dir)
    except Exception as e:
        print(e)
        raise Exception('checkpoint saving failure')


def load_checkpoint(load_dir, map_location=None):
    try:
        print('loading checkpoint from {}'.format(load_dir))
        checkpoint = torch.load(load_dir, map_location=map_location)
        print('loading successful')
        return checkpoint
    except Exception as e:
        print(e)
        print('No checkpoint and begin new training')


def report_num_params(model):
    print('Number of parameters of model: {:.2E}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))


class Model(nn.Module):
    def __init__(self, is_train=True, arg_dir='parameter.json', **kwargs):
        super().__init__()
        try:
            m_param = load_parameters(path='model/models.json')
            arg_params = load_parameters(path=arg_dir)
            print('Parameters loaded')
            print(''.join(['-' for i in range(30)]))
            t_param, v_param, c_param = arg_params['train'], arg_params['test'], arg_params['common']
            if is_train:
                for i in sorted(t_param):
                    print('{:<15s} -> {}'.format(str(i), t_param[i]))
            else:
                for i in sorted(v_param):
                    print('{:<15s} -> {}'.format(str(i), v_param[i]))
        except Exception as e:
            print(e)
            raise ValueError('Parameter not found.')

        self.model_name = c_param['name']
        self.mode = c_param['type']
        self.is_train = is_train
        self.epoch = t_param['begin_epoch'] if self.is_train else 0
        self.num_epoch = t_param['num_epoch'] if is_train else 1
        self.lr = t_param['learning_rate'] * t_param['decay_rate'] ** self.epoch
        self.decay_rate = t_param['decay_rate']
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        print('Using device {}'.format(self.device))
        if self.model_name is None:
            raise Exception('You must define either an upscale model or a downscale model for super resolution')
        if self.mode not in m_param.keys():
            raise ValueError('Wrong mode. Try {}'.format(', '.join(m_param.keys())))
        if self.model_name not in m_param[self.mode]:
            raise ValueError('Wrong model name. Try {}'.format(', '.join(m_param[self.mode].keys())))

        root_dir = c_param['root_dir']
        self.val_hr_dir = os.path.join(root_dir, c_param['s0_dir'], v_param['hr_dir'])
        self.val_lr_dir = os.path.join(root_dir, c_param['s0_dir'], v_param['lr_dir'])
        self.sr_out_dir = os.path.join(root_dir, c_param['s1_dir'], self.model_name, v_param['sr_dir'])
        if not os.path.isdir(self.sr_out_dir):
            os.makedirs(self.sr_out_dir)

        self.log_dir = os.path.join(root_dir, c_param['log_dir'].format(self.model_name))
        self.checkpoint = os.path.join(root_dir, c_param['ckpt_dir'].format(self.model_name))
        self.map_location = t_param['map_location']
        self.metric = PSNR()
        self.t_format = '{:^6s} | {:^6s} | {:^7s} | {:^7s} | {:^7s} | {:^7s} | {:^8s} '
        self.r_format = '{:^6d} | {:^6d} | {:^7.4f} | {:^7.4f} | {:^7.4f} | {:^7.4f} | {:^8.4E} '
        self.t = self.t_format.format('Epoch', 'Batch', 'BLoss', 'ELoss', 'PSNR', 'AVGPSNR', 'Runtime')
        self.splitter = ''.join(['-' for i in range(len(self.t))])

        path = '.'.join(['model', self.mode])
        module = getattr(import_module(path), m_param[self.mode][self.model_name.lower()])
        self.model = module(**kwargs)
        self.model = nn.DataParallel(self.model).cuda()
        report_num_params(self.model)
        self.load_checkpoint()
        self.timer = Timer()

        if not self.is_train:
            print('Disabling auto gradient and switching to TEST mode')
            self.eval()
        else:
            self.train()
            print('{} model is ready for training'.format(self.mode))
            self.train_hr_dir = os.path.join(root_dir, c_param['s0_dir'], t_param['hr_dir'])
            self.train_lr_dir = os.path.join(root_dir, c_param['s0_dir'], t_param['lr_dir'])
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decay_rate)
            train_dataset = SRTrainDataset(hr_dir=self.train_hr_dir, lr_dir=self.train_lr_dir, h=t_param['window'][0],
                                           w=t_param['window'][1], scale=c_param['scale'], num_per=t_param['num_per'])
            self.train_loader = DataLoader(train_dataset, batch_size=t_param['batch_size'], shuffle=True,
                                           num_workers=t_param['num_worker'])
        val_dataset = SRTestDataset(hr_dir=self.val_hr_dir, lr_dir=self.val_lr_dir)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=t_param['num_worker'])

    def load_checkpoint(self, strict=False):
        if self.checkpoint is not None and os.path.isfile(self.checkpoint):
            try:
                print('loading checkpoint from {}'.format(self.checkpoint))
                ckpt = torch.load(self.checkpoint, map_location=self.map_location)
                if ckpt is None:
                    print('No checkpoint and start new training for {} model'.format(self.mode))
                else:
                    print('loading successful and recovering checkpoints for {} model'.format(self.mode))
                    self.load_state_dict(ckpt, strict=strict)
                    print('Checkpoint loaded successfully')
            except:
                print('Checkpoint failed to load, continuing without pretrained checkpoint')
                # raise ValueError('Wrong Checkpoint path or loaded erroneously')
        else:
            print('No checkpoint and start new training for {} model'.format(self.mode))

    def save_checkpoint(self, add_time=False):
        try:
            torch.save(self.state_dict(), self.checkpoint)
        except Exception as e:
            print(e)
            raise Exception('checkpoint saving failed')

    def forward(self, x, clip_round=False):
        output = self.model(x)
        if clip_round:
            output = torch.clamp(torch.round(output), 0., 255.)
        return output

    def train_step(self, loss_fn):
        if not self.is_train:
            raise Exception('Training disabled. Please reinitialize the model.')
        self.train()
        ls, ps = list(), list()
        for bid, batch in enumerate(self.train_loader):
            hr, lr = batch['hr'].cuda(), batch['lr'].cuda()
            self.optimizer.zero_grad()
            sr = self.forward(lr)

            l = loss_fn(hr, sr, lr)
            ls.append(l)
            psnr = self.metric(sr, hr).detach().cpu().item()
            ps.append(psnr)
            print(self.r_format.format(self.epoch, bid, l, sum(ls) / len(ls),
                                       psnr, sum(ps) / len(ps), self.timer.report()))
            print(self.t, end='\r')

            l.backward()
            self.optimizer.step()
            self.timer.refresh()

        self.epoch += 1
        self.scheduler.step()
        with open(self.log_dir, 'a') as f:
            f.write(self.r_format.format(self.epoch, -1, -1.0, sum(ls) / len(ls), -1.0, sum(ps) / len(ps),
                                         self.timer.report()))
            f.write('\n')
        print(self.splitter)

    def test_step(self, loss_fn, self_ensemble=False, save=False):
        self.eval()
        ls, ps = list(), list()
        with torch.no_grad():
            for bid, batch in enumerate(self.val_loader):
                hr, lr = batch['hr'].cuda(), batch['lr'].cuda()
                sr = self.forward(lr)
                if self_ensemble:
                    sr += self.forward(lr.flip(3)).flip(3)
                    sr += self.forward(lr.flip(3).rot90(1, [2, 3])).rot90(3, [2, 3]).flip(3)
                    sr += self.forward(lr.flip(3).rot90(2, [2, 3])).rot90(2, [2, 3]).flip(3)
                    sr += self.forward(lr.flip(3).rot90(3, [2, 3])).rot90(1, [2, 3]).flip(3)
                    sr += self.forward(lr.rot90(1, [2, 3])).rot90(3, [2, 3]) 
                    sr += self.forward(lr.rot90(2, [2, 3])).rot90(2, [2, 3])
                    sr += self.forward(lr.rot90(3, [2, 3])).rot90(1, [2, 3])
                    sr /= 8.0
                psnr = self.metric(sr, hr).detach().cpu().item()
                l = loss_fn(hr, sr, lr).detach().cpu().item()
                ps.append(psnr)
                ls.append(l)
                print(self.r_format.format(-1, bid, l, sum(ls) / len(ls), psnr, sum(ps) / len(ps), self.timer.report()))
                print(self.t, end='\r')
                self.timer.refresh()
                if save:
                    img = torch.clamp(torch.round(sr), 0., 255.).detach().cpu().numpy().astype(np.uint8)
                    img = np.squeeze(np.moveaxis(img, 1, -1), axis=0).astype(np.uint8)
                    imwrite(os.path.join(self.sr_out_dir, *batch['name']), img, format='png', compress_level=0)
        return np.mean(ps)

    def train_model(self, loss_fn):
        print(self.splitter)
        print(self.t)
        print(self.splitter)
        best_val = self.test_step(loss_fn)
        torch.cuda.empty_cache()
        print(self.splitter)
        print('Best-by-far model stays at {:.4f}'.format(best_val))
        print(self.splitter)
        for epoch in range(self.num_epoch):
            self.train_step(loss_fn)
            val_l = self.test_step(loss_fn)
            if best_val is None or best_val < val_l:
                self.save_checkpoint()
                best_val = val_l
                print(self.splitter)
                print('Saving best-by-far model at {:.4f}'.format(best_val))
            print(self.splitter)

    def eval_model(self, loss_fn, self_ensemble=False, save=False):
        print(self.splitter)
        print(self.t)
        print(self.splitter)
        best_val = self.test_step(loss_fn, self_ensemble=self_ensemble, save=save)
        print(self.splitter)
        print('Best-by-far model stays at {:.4f}'.format(best_val))
        print('Images saved to {}'.format(self.sr_out_dir))
        print(self.splitter)
