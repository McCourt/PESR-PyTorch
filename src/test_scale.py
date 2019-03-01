import os, gc
import numpy as np
from skimage.transform import resize
from imageio import imread


for img_name in sorted(os.listdir('imgs/TrainGT/')):
    if not img_name.endswith('png'):
        continue
    hr = imread('../imgs/source_image/TrainGT/{}'.format(img_name))
    lr = imread('../imgs/source_image/TrainLR/{}'.format(img_name))
    h, w, c = hr.shape
    max_mse = None
    for scale in np.linspace(2, 4, num=5):
        for order_1 in range(6):
            for order_2 in range(6):
                mse = np.mean(np.square(
                    resize(
                        resize(
                            hr, (h // scale, w // scale), mode='reflect', preserve_range=True, order=order_1
                        ),
                        (h, w), mode='reflect', preserve_range=True, order=order_2
                    ) - hr
                ))
                if max_mse is None or mse < max_mse:
                    max_mse = mse
                    print('{} {} {} {} {}'.format(img_name, scale, order_1, order_2, max_mse), end='\r')
    print()
    gc.collect()
