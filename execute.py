import getopt
import sys
from imageio import imread, imwrite
from torch.utils.data import DataLoader
from dataset import ResolutionDataset
from helper import *

if __name__ == '__main__':
    args = sys.argv[1:]
    longopts = [
        'mode=', 'model-name=', 'hr-dir=',
        'lr-dir=', 'output-dir=', 'upsample=',
        'window=', 'learning-rate=', 'decay-rate=',
        'decay-step=',  'batch-size=', 'num-epoch=',
        'add-dsloss=', 'gpu-device=', 'log-dir=',
        'ckpt-dir=', 'print-every=', 'attention='
    ]

    try:
        optlist, args = getopt.getopt(args, '', longopts)
    except getopt.GetoptError as e:
        print(str(e))
        sys.exit(2)
    for arg, opt in optlist:
        if arg == '--mode':
            MODE = str(opt)
        elif arg == '--model-name':
            MODEL_NAME = str(opt)
            model = load_model(MODEL_NAME)
        elif arg == '--hr-dir':
            HR_DIR = str(opt)
            if not os.path.exists(HR_DIR) or len(os.listdir(HR_DIR)) == 0:
                raise Exception('High resolution images can not find. HR path invalid.')
        elif arg == '--lr-dir':
            LR_DIR = str(opt)
            if not os.path.exists(LR_DIR) or len(os.listdir(LR_DIR)) == 0:
                raise Exception('Low resolution images can not find. LR path invalid.')
        elif arg == '--output-dir':
            OUT_DIR = str(opt)
            if not os.path.exists(OUT_DIR):
                os.mkdir(OUT_DIR)
            if not os.path.exists(os.path.join(OUT_DIR, MODEL_NAME)):
                os.mkdir(os.path.join(OUT_DIR, MODEL_NAME))
        elif arg == '--log-dir':
            LOG_PATH = str(opt)
            if not os.path.exists(LOG_PATH):
                os.mkdir(LOG_PATH)
        elif arg == '--ckpt-dir':
            CKPT_DIR = str(opt)
            if not os.path.exists(CKPT_DIR):
                os.mkdir(CKPT_DIR)
        elif arg == '--learning-rate':
            LEARNING_RATE = float(opt)
        elif arg == '--decay-rate':
            DECAY_RATE = float(opt)
        elif arg == '--decay-step':
            DECAY_STEP = int(opt)
        elif arg == '--batch-size':
            BATCH_SIZE = int(opt)
        elif arg == '--num-epoch':
            NUM_EPOCH = int(opt)
        elif arg == '--window':
            WIN_SIZE = int(opt)
        elif arg == '--gpu-device':
            DEVICE = torch.device(str(opt) if torch.cuda.is_available else 'cpu')
            MAP_LOC = None
            if str(opt) == 'cuda':
                MAP_LOC = {
                    'cuda:0':'cuda',
                    'cuda:1':'cuda',
                    'cuda:2':'cuda',
                    'cuda:3':'cuda'
                }
        elif arg == '--add-dsloss':
            ADD_DS = False if str(opt).lower() == 'false' else True
        elif arg == '--upsample':
            UPSAMPLE = False if str(opt).lower() == 'false' else True
        elif arg == '--attention':
            ATTENTION = False if str(opt) == 'false' else True
        elif arg == '--print-every':
            PRINT_EVERY = int(opt)
        else:
            continue

    print('Initializing model and data')
    if MODE == 'train':
        model = model.to(DEVICE)
        loss = MSEnDSLoss(add_ds=ADD_DS).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=DECAY_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_STEP)
        data_set = ResolutionDataset(HR_DIR, LR_DIR, upscale=UPSAMPLE, h=WIN_SIZE, w=WIN_SIZE)
        data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

        ## load checkpoint
        ckpt = load_checkpoint(load_dir=CKPT_DIR, map_location=MAP_LOC, model_name=MODEL_NAME)
        if ckpt is not None:
            print('recovering from checkpoints...')
            model.load_state_dict(ckpt['model'])
            begin_epoch = ckpt['epoch'] + 1
            print('resuming training')
        else:
            begin_epoch = 0

        timer = Timer()
        with open(os.path.join(LOG_PATH, '{}.log'.format(MODEL_NAME)), 'a') as f:
            for epoch in range(begin_epoch, NUM_EPOCH):
                epoch_loss = 0
                for bid, batch in enumerate(data_loader):
                    hr, lr = batch['hr'].to(DEVICE), batch['lr'].to(DEVICE)
                    optimizer.zero_grad()
                    x = model(lr)
                    l = loss(x, hr, lr)
                    l.backward()
                    optimizer.step()
                    epoch_loss += l

                    bpsnr = psnr(l)
                    epsnr = psnr(epoch_loss / (1 + bid))
                    r = report(epoch, bid, l, epoch_loss, bpsnr, epsnr, timer.report())
                    if bid % PRINT_EVERY == 0:
                        print(r)
                    f.write(r)
                    f.flush()
                state_dict = {
                    'model': model.state_dict(),
                    'epoch': epoch
                }
                save_checkpoint(state_dict, CKPT_DIR, model_name=MODEL_NAME)
    elif MODE == 'output':
        model = model.to(DEVICE)
        print('Outputting with {}'.format(MODEL_NAME))
        ckpt = load_checkpoint(load_dir=CKPT_DIR, model_name=MODEL_NAME, map_location=MAP_LOC)
        if ckpt is not None:
            print('recovering from checkpoints...')
            model.load_state_dict(ckpt['model'])
            for param in model.parameters():
                param.requires_grad = False
            print('loading successful and begin outputting')
        else:
            raise Exception('Checkpoint not found. Check ckpt_path.')
        psnrs = []
        for IMG_NAME in sorted(os.listdir(LR_DIR)):
            lr_img = np.array(imread(os.path.join(LR_DIR, IMG_NAME))).astype(np.float32)
            hr_img = np.array(imread(os.path.join(HR_DIR, IMG_NAME))).astype(np.float32)
            h, w, c = lr_img.shape
            sr_img = np.zeros(shape=(4 * h, 4 * w, c))
            for i in range(0, h, WIN_SIZE):
                for j in range(0, w, WIN_SIZE):
                    win_lr = lr_img[i:i + WIN_SIZE, j:j + WIN_SIZE, :]
                    win_lr_img = np.moveaxis(win_lr, -1, 0)
                    lr_tensor = torch.tensor(win_lr_img).to(DEVICE)
                    sr_tensor = torch.clamp(torch.round(model(lr_tensor)), 0., 255.)
                    win_sr_img = sr_tensor.detach().cpu().numpy().astype(np.uint8)
                    sr_img[i * 4:(i + WIN_SIZE) * 4, j * 4:(j + WIN_SIZE) * 4, :] = np.moveaxis(win_sr_img, 0, -1)
            try:
                imwrite(os.path.join(OUT_DIR, MODEL_NAME, IMG_NAME), sr_img, format='png', compress_level=0)
                img_psnr = 10 * torch.log10(255 ** 2 / np.mean(np.square(sr_img - hr_img)))
                print('PSNR {:.6f} | Image {}'.format(img_psnr, IMG_NAME))
            except:
                continue
    elif MODE == 'test':
        pass
    else:
        raise Exception('Invalid Mode. Please select between train, output and test.')