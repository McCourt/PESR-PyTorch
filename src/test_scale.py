import os, gc
import numpy as np
from skimage.transform import resize
from imageio import imread
import matplotlib.pyplot as plt


for img_name in sorted(os.listdir('../imgs/source_image/TrainGT/')):
    if not img_name.endswith('png'):
        continue
    hr = imread('../imgs/source_image/TrainGT/{}'.format(img_name))
    lr = imread('../imgs/source_image/TrainLR/{}'.format(img_name))
    h, w, c = hr.shape
    max_mse = None
    dic = dict()
    scales = np.linspace(2, 4, num=100)
    for order_1 in range(6):
        dic[order_1] = []
        order_2 = 3
        for scale in scales:
            mse = np.mean(np.square(
                resize(
                    resize(
                        hr, (h // scale, w // scale), mode='reflect', preserve_range=True, order=order_1
                    ),
                    (h, w), mode='reflect', preserve_range=True, order=order_2
                ) - hr
            ))
            dic[order_1].append(mse)
            # if max_mse is None or mse < max_mse:
            #     max_mse = mse
            #     print('{} {} {} {} {}'.format(img_name, scale, order_1, order_2, max_mse), end='\r')
    for i in dic:
        print("plot {}".format(i))
        plt.plot(scales, dic[i], label=i)
    plt.legend()
    # plt.savefig('./scale/{}'.format(img_name))
    # plt.close()
    plt.show()
    gc.collect()
    break
