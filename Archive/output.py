import numpy as np
import sys
import getopt
from imageio import imread, imwrite
from src.helper import *

if __name__ == '__main__':
    args = sys.argv[1:]
    longopts = [
        'model-name=', 'lr-dir=', 'output-dir=', 'old-loc=',
        'upsample=', 'gpu-device=', 'ckpt-dir=', 'hr-dir='
    ]

    try:
        optlist, args = getopt.getopt(args, '', longopts)
    except getopt.GetoptError as e:
        print(str(e))
        sys.exit(2)
    for arg, opt in optlist:
        if arg == '--model-name':
            model_name = str(opt)
            model = load_model(model_name)
        elif arg == '--lr-dir':
            LR_DIR = str(opt)
        elif arg == '--hr-dir':
            HR_DIR = str(opt)
        elif arg == '--output-dir':
            OUT_DIR = str(opt)
            if not os.path.exists(OUT_DIR):
                os.mkdir(OUT_DIR)
            if not os.path.exists(os.path.join(OUT_DIR, model_name)):
                os.mkdir(os.path.join(OUT_DIR, model_name))
        elif arg == '--gpu-device':
            DEVICE = torch.device(str(opt) if torch.cuda.is_available else 'cpu')
        elif arg == '--old-loc':
            map_location={'cuda:0':'cuda', 'cuda:1':'cuda', 'cuda:2':'cuda', 'cuda:3':'cuda'}
        elif arg == '--ckpt-dir':
            CKPT_DIR = str(opt)
        elif arg == '--upsample':
            UPSAMPLE = False if str(opt).lower() == 'false' else True
        else:
            continue

    model = model.to(DEVICE)
    print(model_name)
    ckpt = load_checkpoint(load_dir=CKPT_DIR, model_name=model_name, map_location=map_location)
    if ckpt is not None:
        print('recovering from checkpoints...')
        model.load_state_dict(ckpt['model'])
        for param in model.parameters():
            param.requires_grad = False
        print('loading successful and begin outputing')
    psnrs = []
    for IMG_NAME in sorted(os.listdir(LR_DIR)):
        try:
            img = np.expand_dims(np.moveaxis(imread(os.path.join(LR_DIR, IMG_NAME))[:, :, :], -1, 0), axis=0).astype(np.float32)
            #img = np.moveaxis(imread(os.path.join(LR_DIR, IMG_NAME)), -1, 0).astype(np.float32)
            img_tensor = torch.tensor(img).to(DEVICE)
            img_tensor = torch.clamp(torch.round(model(img_tensor)), 0., 255.)
            # hr = np.moveaxis(imread(os.path.join(HR_DIR, IMG_NAME)), -1, 0).astype(np.float32)
            hr = np.expand_dims(np.moveaxis(imread(os.path.join(HR_DIR, IMG_NAME))[:, :, :], -1, 0), axis=0).astype(np.float32)
            hr_tensor = torch.tensor(hr).to(DEVICE)
            output = img_tensor.detach().cpu().numpy().astype(np.uint8)
            output = output.reshape(output.shape[1:])
            imwrite(os.path.join(OUT_DIR, model_name, IMG_NAME), np.moveaxis(output, 0, -1), format='png', compress_level=0)
            print('Image {} | PSNR {:.6f}'.format(IMG_NAME, psnr(nn.MSELoss()(img_tensor, hr_tensor))))
        except:
            continue
