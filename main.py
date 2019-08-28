import torch
from loss import DSLoss as Loss
from model import Model
import sys
import getopt
from datetime import datetime
from src.helper import report_time


scale = 32
ds_weight = 0.0
is_new = False
self_ensemble = False
save_img = False

is_new = True
# self_ensemble = True
# save_img = True
num_groups=32
rep_pad=False


if __name__ == '__main__':
    print('----->>>{}<<<-----'.format('{} GPUs Available'.format(torch.cuda.device_count())))

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
    model = Model(scale=scale, is_train=is_train, num_blocks=32, num_channel=256) # EDSR
    # model = Model(scale=scale, is_train=is_train, num_groups=12, num_channel=128, rep_pad=False) # DSSR
    loss = Loss(scale=scale, weight=ds_weight)
    if is_train:
        try:
            model.train_model(loss_fn=loss, new=is_new)
        except KeyboardInterrupt:
            save = input("\n{}: Interrupted and save model? (y/n)".format(report_time()))
            assert(save in ['y', 'n'], 'Invalid input')
            if save == 'y':
                model.save_checkpoint(add_time=True)
            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(-1)
    else:
        model.eval_model(loss_fn=loss, self_ensemble=self_ensemble, save=save_img)
