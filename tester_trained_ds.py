from downsample.conv import ConvolutionDownscale
import torch
from torch import nn
from imageio import imread, imwrite
import numpy as np
import sys, os
import getopt
from time import time
from helper import psnr
from skimage.color import gray2rgb
from skimage.measure import compare_psnr
from helper import load_checkpoint

if __name__=='__main__':
    args = sys.argv[1:]
    longopts = [
        'model-name=', 'image-list=', 'hr-dir=', 'lr-dir=', 'ckpt-dir=',
        'base-dir=', 'output-dir=', 'learning-rate=', 'num-epoch=',
        'gpu-device=', 'log-dir=', 'print-every=', 'attention=',
         'clip=', 'scale=', 'lambda=', 'save='
    ]
    try:
        optlist, args = getopt.getopt(args, '', longopts)
    except getopt.GetoptError as e:
        print(str(e))
        sys.exit(2)
    for arg, opt in optlist:
        if arg == '--model-name':
            model_name = str(opt)
        elif arg == '--image-list':
            IMG_LIST = sorted([i for i in os.listdir(str(opt))])
        elif arg == '--hr-dir':
            HR_DIR = str(opt)
        elif arg == '--lr-dir':
            LR_DIR = str(opt)
        elif arg == '--base-dir':
            BASE_DIR = str(opt)
        elif arg == '--output-dir':
            OUT_DIR = str(opt)
        elif arg == '--learning-rate':
            LEARING_RATE = float(opt)
        elif arg == '--num-epoch':
            NUM_EPOCH = int(opt)
        elif arg == '--ckpt-dir':
            CKPT_DIR = str(opt)
        elif arg == '--gpu-device':
            DEVICE = torch.device(str(opt) if torch.cuda.is_available else 'cpu')
        elif arg == '--log-dir':
            LOG_PATH = str(opt)
        elif arg == '--print-every':
            PRINT_EVERY = int(opt)
        elif arg == '--attention':
            ATTENTION = False if str(opt) == 'false' else True
        elif arg == '--scale':
            SCALE = int(opt)
        elif arg == '--clip':
            CLIP = int(opt)
        elif arg == '--lambda':
            LAMBDA = float(opt)
        elif arg == '--save':
            SAVE = False if str(opt) == 'false' else True

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    if not os.path.exists(os.path.join(OUT_DIR, 'x{}'.format(SCALE))):
        os.mkdir(os.path.join(OUT_DIR, 'x{}'.format(SCALE)))
    OUT_DIR = os.path.join(OUT_DIR, 'x{}'.format(SCALE))

    if not os.path.exists(os.path.join(OUT_DIR, 'sr')):
        os.mkdir(os.path.join(OUT_DIR, 'sr'))
    if not os.path.exists(os.path.join(OUT_DIR, 'dsr')):
        os.mkdir(os.path.join(OUT_DIR, 'dsr'))

    attentioner = nn.Sigmoid()
    bds = ConvolutionDownscale()
    ckpt = load_checkpoint(load_dir='./checkpoints/', map_location=None, model_name='down_sample')
    try:
        if ckpt is not None:
            print('recovering from checkpoints...')
            bds.load_state_dict(ckpt['model'])
            print('resuming training')
    except:
        raise FileNotFoundError('Check checkpoints')

    lr_loss = nn.MSELoss()
    l2_loss = nn.MSELoss()
    hr_loss = nn.MSELoss()
    lambdas = np.linspace(0, 1, 101)
    with open(os.path.join(LOG_PATH, '{}.log'.format(model_name)), 'w') as f:
        for IMG_NAME in IMG_LIST:
            sr_name = IMG_NAME.replace('.', '.')

            lr_img = np.array(imread(os.path.join(LR_DIR, IMG_NAME)))
            sr_img = np.array(imread(os.path.join(BASE_DIR, sr_name)))
            hr_img = np.array(imread(os.path.join(HR_DIR, IMG_NAME)))

            if len(lr_img.shape) == 2:
                lr_img = gray2rgb(lr_img)
            if len(sr_img.shape) == 2:
                sr_img = gray2rgb(sr_img)
            if len(hr_img.shape) == 2:
                hr_img = gray2rgb(hr_img)

            h, w = min(sr_img.shape[0], hr_img.shape[0]), min(sr_img.shape[1], hr_img.shape[1])
            sr_img = sr_img[:h, :w, :]
            hr_img = hr_img[:h, :w, :]
            # ssim_dict[IMG_NAME].append(compare_ssim(sr_img[CLIP:-CLIP, CLIP:-CLIP, :],
            #                                        hr_img[CLIP:-CLIP, CLIP:-CLIP, :]))

            lr_tensor = torch.tensor(np.expand_dims(lr_img.astype(np.float32), axis=0)).type('torch.DoubleTensor').to(DEVICE)
            in_tensor = torch.tensor(np.expand_dims(sr_img.astype(np.float32), axis=0)).type('torch.DoubleTensor').to(DEVICE)
            org_tensor = torch.tensor(np.expand_dims(sr_img.astype(np.float32), axis=0)).type('torch.DoubleTensor').to(DEVICE)
            in_tensor.requires_grad = True
            psnrs = []
            begin_time = time()
            try:
                for epoch in range(NUM_EPOCH):
                    ds_in_tensor = bds(in_tensor, nhwc=True)
                    lr_l = lr_loss(ds_in_tensor, lr_tensor)
                    l2_l = l2_loss(in_tensor, org_tensor)
                    l = lr_l + LAMBDA * l2_l
                    l.backward()

                    gradient = in_tensor.grad * LEARING_RATE
                    if ATTENTION:
                        gradient = gradient * attentioner(torch.abs(gradient) / torch.max(torch.abs(gradient)))
                    in_tensor = in_tensor.data.sub_(gradient)

                    in_tensor.requires_grad = True

                    sr_img = torch.clamp(torch.round(in_tensor), 0., 255.).detach().cpu().numpy().astype(np.uint8)
                    # sr_img = np.moveaxis(sr_img.reshape(np_ds.shape[1:]), 0, -1)
                    report = '{} | {:.4f} | {:.4f}'.format(IMG_NAME, lr_l, l2_l, compare_psnr(sr_img, hr_img), time() - begin_time)
                    if epoch == 0 or epoch == NUM_EPOCH - 1:
                        print(report)
                    f.write(report)
                # psnrs.append(float(psnr(hr_l)))
                    # sr_img = torch.clamp(torch.round(in_tensor), 0., 255.).detach().cpu().numpy().astype(np.int16)
                    # sr_img = rgb2ycbcr(sr_img.reshape(sr_img.shape[1:]))[CLIP:-CLIP, CLIP:-CLIP, 0]
                    # psnr_dict[IMG_NAME].append(float(psnr(hr_l)))
                    # psnr_dict[IMG_NAME].append(10 * np.log10(219 ** 2 / np.mean(np.square(sr_img - hr_img))))
                    # ssim_dict[IMG_NAME].append(compare_ssim(sr_img[CLIP:-CLIP, CLIP:-CLIP, :].astype(np.int16),
                    #                                         hr_img[CLIP:-CLIP, CLIP:-CLIP, :].astype(np.int16),
                    #                                         data_range=255, multichannel=True))

                if SAVE:
                    sr_img = torch.clamp(torch.round(in_tensor), 0., 255.).detach().cpu().numpy().astype(np.uint8)
                    # sr_img = sr_img.reshape(np_ds.shape[1:])
                    imwrite(os.path.join(OUT_DIR, 'sr', IMG_NAME), sr_img, format='png', compress_level=0)
                    lr_img = torch.clamp(torch.round(ds_in_tensor), 0., 255.).detach().cpu().numpy().astype(np.uint8)
                    # lr_img = lr_img.reshape(np_ds.shape[1:])
                    imwrite(os.path.join(OUT_DIR, 'dsr', IMG_NAME), lr_img, format='png', compress_level=0)

                # rt = time() - begin_time
                # cnt += 1
                # prior_psnr = psnr_dict[IMG_NAME][0]
                # post_psnr = psnr_dict[IMG_NAME][-1]
                # diff_psnr = post_psnr - prior_psnr
                # mean_diff_psnr += diff_psnr
                # mean_prior_psnr += prior_psnr
                # mean_post_psnr += post_psnr
                # print('{} | PRIOR PSNR {:.8f} | POST PSNR {:.8f} | PSNR CHANGE {:.8f} | RT {:.4f}'.format(IMG_NAME, prior_psnr, post_psnr, diff_psnr, rt))

                #prior_ssim = ssim_dict[IMG_NAME][0]
                #post_ssim = ssim_dict[IMG_NAME][-1]
                #diff_ssim = post - prior
                #mean_diff_ssim += diff_ssim
                #mean_prior_ssim += prior_ssim
                #mean_post_ssim += post_ssim
                #print('{} | PRIOR PSNR {:.8f} | POST PSNR {:.8f} | PSNR CHANGE {:.8f} | RT {:.4f}'.format(IMG_NAME, prior_ssim, post_ssim, diff_ssim, rt))
            except Exception as e:
                print(e)
                print('Failure on {}'.format(IMG_NAME))
        print(np.mean(psnrs))
        # print('Mean prior PNSR {}\nMean posterior PSNR {}\nMean increase of PSNR {}'.format(mean_prior_psnr / cnt, mean_post_psnr / cnt, mean_diff_psnr / cnt))
        #print('Mean prior SSIM {}\nMean posterior SSIM {}\nMean increase of SSIM {}'.format(mean_prior_ssim / cnt, mean_post_ssim / cnt, mean_diff_ssim / cnt))
