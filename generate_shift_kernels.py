import glob
from multiprocessing import Pool

import numpy as np
import tqdm
import scipy.ndimage
import os
from matplotlib import pyplot as plt
from PIL import Image
import MeanShift
import imresize as im
import torch

HR = sorted(glob.glob("DIFF/TRAIN_HR/0001.png"))

def shift(a,i,j):
    return np.roll(a, (i, j), axis=(0, 1))

def crop(a):
    return a[50:-50,50:-50,:]

SIZE=10
X=[]
Y1=[]
Y2=[]

for files in HR:
    print(files)
    hr = Image.open(files)
    hr = np.array(hr)/255
    lr = im.imresize(hr,output_shape=(hr.shape[0] // 4, hr.shape[1] // 4))

    lrshift1 = shift(hr,1,0)
    lrshift1 = im.imresize(lrshift1,output_shape=(lrshift1.shape[0] // 4, lrshift1.shape[1] // 4))
    lrshift1 = crop(lrshift1)

    lrshift2 = shift(hr, 2, 0)
    lrshift2 = im.imresize(lrshift2, output_shape=(lrshift2.shape[0] // 4, lrshift2.shape[1] // 4))
    lrshift2 = crop(lrshift2)

    kernelRange = range(-SIZE,SIZE+1)
    A = np.asarray([crop(shift(lr,i,0)).flatten() for i in kernelRange])
    X+=[A]
    Y1 +=[lrshift1.flatten()]
    Y2 += [lrshift2.flatten()]

X=np.concatenate(X,1)
X=np.transpose(X)
Y1=np.concatenate(Y1)
Y1=np.concatenate(Y2)

kernel0 = np.zeros((2*SIZE+1,))
kernel0[SIZE] = 1
kernel1 = np.linalg.lstsq(X,Y1)[0]
kernel2 = np.linalg.lstsq(X,Y1)[0]
kernels=[kernel0,kernel1,kernel2]

dictionary={}
for i in range(0,3):
    for j in range(0,3):
        conv_kernel = torch.zeros(3,3,2*SIZE+1,2*SIZE+1)
        for k in range(3):
            conv_kernel[k,k,:,:] = torch.DoubleTensor(np.outer(kernels[i],kernels[j]))
        dictionary[(i,j)] = conv_kernel
        if i != 0:
            dictionary[(-i, j)] = torch.flip(dictionary[(i,j)],(3,))
        if (j != 0):
            dictionary[(i, -j)] = torch.flip(dictionary[(i, j)], (2,))
        if (i != 0 and j != 0):
            dictionary[(-i, -j)] = torch.flip(dictionary[(i, j)], (2, 3))

torch.save(dictionary,'shifts10.pt')

