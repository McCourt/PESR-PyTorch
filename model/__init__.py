import torch
import torch.nn as nn
from importlib import import_module
from src.helper import load_parameters, Timer, fourier_transform, output_report, report_time
from loss.psnr import PSNR
from loss.ssim import SSIM
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
        output_report('loading checkpoint from {}'.format(load_dir))
        checkpoint = torch.load(load_dir, map_location=map_location)
        output_report('loading successful')
        return checkpoint
    except Exception as e:
        print(e)
        output_report('No checkpoint and begin new training')


def report_num_params(model):
    output_report(
        'Number of parameters of model: {:.2E}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))


class Model(nn.Module):
    def __init__(self, scale, is_train=True, arg_dir='parameter.json', **kwargs):
        super().__init__()
        try:
            m_param = load_parameters(path='model/models.json')
            arg_params = load_parameters(path=arg_dir)
            print('+' + ''.join(['-' for i in range(30)]) + '+')
            print('|{:^30}|'.format('Parameters loaded'))
            print('+' + ''.join(['-' for i in range(30)]) + '+')
            t_param, v_param, c_param = arg_params['train'], arg_params['test'], arg_params['common']
            self.model_name = c_param['name']
            self.scale = scale  # c_param['scale']
            self.mode = c_param['type']
            print('|{:<15s} -> {:<11s}|'.format('model', self.model_name))
            print('|{:<15s} -> {:<11s}|'.format('scale', str(self.scale)))
            print('|{:<15s} -> {:<11s}|'.format('model_type', self.mode))
            if is_train:
                for i in sorted(t_param):
                    if 'dir' not in str(i): print('|{:<15s} -> {:<11s}|'.format(str(i), str(t_param[i])))
            else:
                for i in sorted(v_param):
                    if 'dir' not in str(i): print('|{:<15s} -> {:<11s}|'.format(str(i), str(v_param[i])))
            print('+' + ''.join(['-' for i in range(30)]) + '+')
        except Exception as e:
            print(e)
            raise ValueError('Parameter not found.')

        self.is_train = is_train
        self.trim = c_param['trim']
        self.epoch = t_param['begin_epoch'] if self.is_train else 0
        self.num_epoch = t_param['num_epoch'] if is_train else 1
        self.lr = t_param['learning_rate'] * t_param['decay_rate'] ** self.epoch
        self.decay_rate = t_param['decay_rate']
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        output_report('Using device {}'.format(self.device))
        if self.model_name is None:
            raise Exception('You must define either an upscale model or a downscale model for super resolution')
        if self.mode not in m_param.keys():
            raise ValueError('Wrong mode. Try {}'.format(', '.join(m_param.keys())))
        if self.model_name not in m_param[self.mode]:
            raise ValueError('Wrong model name. Try {}'.format(', '.join(m_param[self.mode].keys())))

        root_dir = c_param['root_dir']
        # self.val_hr_dir = os.path.join(root_dir, c_param['s0_dir'], v_param['hr_dir'].format(self.scale))
        # self.val_lr_dir = os.path.join(root_dir, c_param['s0_dir'], v_param['lr_dir'].format(self.scale))
        self.sr_out_dir = os.path.join(root_dir, c_param['s1_dir'], self.model_name,
                                       v_param['sr_dir'].format(v_param['dataset'], self.scale))
        if not os.path.isdir(self.sr_out_dir):
            os.makedirs(self.sr_out_dir)

        self.log_dir = os.path.join(root_dir, c_param['log_dir'].format(self.model_name, self.scale))
        self.checkpoint = os.path.join(root_dir, c_param['ckpt_dir'].format(self.model_name, self.scale))
        self.map_location = t_param['map_location']
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.t_format = '{:^6s} | {:^6s} | {:^7s} | {:^7s} | {:^7s} | {:^7s} | {:^8s} | {}'
        self.r_format = '{:^6d} | {:^6d} | {:^7.4f} | {:^7.4f} | {:^7.4f} | {:^7.4f} | {:^.2E} | {}'
        self.t = self.t_format.format('Epoch', 'Batch', 'BLoss', 'ELoss', 'PSNR', 'AVGPSNR', 'Runtime',
                                      'Model:{}/Scale:{}'.format(self.model_name, self.scale))
        self.splitter = ''.join(['-' for i in range(len(self.t))])
        self.refresher = ''.join([' ' for i in range(len(self.t))])

        path = '.'.join(['model', self.mode])
        try:
            module = getattr(import_module(path), m_param[self.mode][self.model_name.lower()])
        except:
            try:
                module = getattr(import_module(path), self.model_name)
            except:
                raise Exception('Can not find the model {}'.format(self.model_name))
        self.model = module(scale=self.scale, **kwargs)
        self.model = nn.DataParallel(self.model).cuda()
        report_num_params(self.model)
        self.load_checkpoint()
        self.timer = Timer()

        if not self.is_train:
            output_report('Disabling auto gradient and switching to TEST mode'.format())
            self.eval()
            output_report('{} model is ready for testing'.format(self.mode))
        else:
            self.train()
            output_report('{} model is ready for training'.format(self.mode))
            # self.train_hr_dir = os.path.join(root_dir, c_param['s0_dir'], t_param['hr_dir'].format(self.scale))
            # self.train_lr_dir = os.path.join(root_dir, c_param['s0_dir'], t_param['lr_dir'].format(self.scale))
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decay_rate)
            train_dataset = SRTrainDataset(dataset=t_param['dataset'], scale=self.scale, h=t_param['window'][0],
                                           w=t_param['window'][1], num_per=t_param['num_per'])
            self.train_loader = DataLoader(train_dataset, batch_size=t_param['batch_size'], shuffle=True,
                                           num_workers=t_param['num_worker'])
        val_dataset = SRTestDataset(dataset=v_param['dataset'], scale=self.scale)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=t_param['num_worker'])

    def load_checkpoint(self, strict=True):
        if self.checkpoint is not None and os.path.isfile(self.checkpoint):
            try:
                output_report('loading checkpoint from {}'.format(self.checkpoint))
                ckpt = torch.load(self.checkpoint, map_location=self.map_location)
                if ckpt is None:
                    output_report('No checkpoint and start new training for {} model'.format(self.mode))
                else:
                    output_report('loading successful and recovering checkpoints for {} model'.format(self.mode))
                    self.load_state_dict(ckpt, strict=strict)
                    output_report('Checkpoint loaded successfully')
            except:
                output_report('Checkpoint failed to load, continuing without pretrained checkpoint')
                # raise ValueError('Wrong Checkpoint path or loaded erroneously')
        else:
            output_report('No checkpoint and start new training for {} model'.format(self.mode))

    def save_checkpoint(self, add_time=False):
        try:
            if add_time:
                torch.save(self.state_dict(),
                           '{}_{}.ckpt'.format(self.checkpoint.replace('.ckpt', ''), report_time()).replace(' ', '_'))
            else:
                torch.save(self.state_dict(), self.checkpoint)
            output_report('checkpoint saving succeeded')
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
            # hr, lr = fourier_transform(batch['hr']).cuda(), fourier_transform(batch['lr']).cuda()
            self.optimizer.zero_grad()
            sr = self.forward(lr)

            l = loss_fn(hr, sr, lr)
            ls.append(l)
            psnr = self.psnr(sr, hr, scale=self.scale, trim=self.trim).detach().cpu().item()
            ps.append(psnr)
            print(self.refresher, end='\r')
            print(self.r_format.format(self.epoch, bid, l, sum(ls) / len(ls),
                                       psnr, sum(ps) / len(ps), self.timer.report(), 'NaName'))
            print(self.t, end='\r')

            l.backward()
            self.optimizer.step()
            self.timer.refresh()

        self.epoch += 1
        self.scheduler.step()
        with open(self.log_dir, 'a') as f:
            f.write(self.r_format.format(self.epoch, -1, -1.0, sum(ls) / len(ls), -1.0, sum(ps) / len(ps),
                                         self.timer.report(), 'NaN'))
            f.write('\n')
        print(self.splitter)

    def test_step(self, loss_fn, self_ensemble=False, save=False):
        self.eval()
        ls, ps, ss = list(), list(), list()
        with torch.no_grad():
            for bid, batch in enumerate(self.val_loader):
                hr, lr = batch['hr'].cuda(), batch['lr'].cuda()
                self.timer.refresh()
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
                psnr = self.psnr(sr, hr, scale=self.scale, trim=self.trim).detach().cpu().item()
                ssim = self.ssim(sr, hr).detach().cpu().item()
                l = loss_fn(hr, sr, lr).detach().cpu().item()
                ps.append(psnr)
                ss.append(ssim)
                ls.append(l)
                print(self.refresher, end='\r')
                print(self.r_format.format(-1, bid, l, sum(ls) / len(ls), psnr,  # sum(ss) / len(ss),
                                           sum(ps) / len(ps), self.timer.report(), *batch['name']))
                print(self.t, end='\r')
                if save:
                    name = batch['name'][0]
                    if self_ensemble: name = name.replace('.png', '_SE.png')
                    img = torch.clamp(torch.round(sr), 0., 255.).detach().cpu().numpy().astype(np.uint8)
                    # img = torch.round(sr).detach().cpu().numpy().astype(np.uint8)
                    img = np.squeeze(np.moveaxis(img, 1, -1), axis=0).astype(np.uint8)
                    imwrite(os.path.join(self.sr_out_dir, name), img, format='png', compress_level=0)
        return np.mean(ps)

    def train_model(self, loss_fn, new=False):
        print(self.splitter)
        print(self.t)
        print(self.splitter)
        best_val = self.test_step(loss_fn) if not new else 0
        if new:
            self.epoch = 0
        torch.cuda.empty_cache()
        print(self.splitter)
        output_report('Best-by-far model stays at {:.4f}'.format(best_val))
        print(self.splitter)
        for epoch in range(self.num_epoch):
            self.train_step(loss_fn)
            val_l = self.test_step(loss_fn)
            print(self.splitter)
            if best_val is None or best_val < val_l:
                self.save_checkpoint()
                # self.save_checkpoint(add_time=True)
                best_val = val_l
                output_report('Saving best-by-far model at {:.4f}'.format(best_val))
            else:
                self.save_checkpoint(add_time=True)
                output_report('Best-by-far model stays at {:.4f}'.format(best_val))
            print(self.splitter)

    def eval_model(self, loss_fn, self_ensemble=False, save=False):
        print(self.splitter)
        print(self.t)
        print(self.splitter)
        best_val = self.test_step(loss_fn, self_ensemble=self_ensemble, save=save)
        print(self.splitter)
        output_report('Best-by-far model stays at {:.4f}'.format(best_val))
        if save: output_report('Images saved to {}'.format(self.sr_out_dir))
        print(self.splitter)


class TTOptimizor(nn.Module):
    def __init__(self, scale, is_train=True, arg_dir='parameter.json', **kwargs):
        super().__init__()
        try:
            m_param = load_parameters(path='model/models.json')
            arg_params = load_parameters(path=arg_dir)
            print('+' + ''.join(['-' for i in range(30)]) + '+')
            print('|{:^30}|'.format('Parameters loaded'))
            print('+' + ''.join(['-' for i in range(30)]) + '+')
            t_param, c_param = arg_params['tto'], arg_params['common']
            self.model_name = c_param['name']
            self.scale = scale
            self.mode = c_param['type']
            print('|{:<15s} -> {:<11s}|'.format('model', self.model_name))
            print('|{:<15s} -> {:<11s}|'.format('scale', str(self.scale)))
            print('|{:<15s} -> {:<11s}|'.format('model_type', self.mode))
            for i in sorted(t_param):
                if 'dir' not in str(i): print('|{:<15s} -> {:<11s}|'.format(str(i), str(t_param[i])))
            print('+' + ''.join(['-' for i in range(30)]) + '+')
        except Exception as e:
            print(e)
            raise ValueError('Parameter not found.')
