import torch
from loss import DownScaleLoss
from model import Model
import sys
import getopt

scale = 8
w = 1.
new = True
se = False
save = False

if __name__ == '__main__':
    print('{} GPUs Available'.format(torch.cuda.device_count()))

    # Load system arguments
    args = sys.argv[1:]
    long_opts = ['mode=']

    try:
        optlist, args = getopt.getopt(args, '', long_opts)
    except getopt.GetoptError as e:
        print(str(e))
        sys.exit(2)

    for arg, opt in optlist:
        if arg == '--mode':
            mode = str(opt)
            if mode not in ['train', 'test']:
                raise Exception('Wrong pipeline mode detected')
            is_train = mode == 'train'
        else:
            raise Exception("Redundant Argument Detected")

    # Define and train model
    model = Model(is_train=is_train)
    loss = DownScaleLoss(scale=scale, weight=w)
    if is_train:
        model.train_model(loss_fn=loss, new=new)
    else:
        model.eval_model(loss_fn=loss, self_ensemble=False, save=False)
