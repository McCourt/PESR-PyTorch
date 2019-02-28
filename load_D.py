import torch
from model.discriminator.Discriminator_VGG import Discriminator_VGG_128
from imageio import imread
import numpy as np

ckpt = torch.load("./50000_D.pth")
model = Discriminator_VGG_128()
model.load_state_dict(ckpt)
hr = imread('imgs/hr.png')
model(torch.Tensor(np.moveaxis(np.expand_dims(hr, 0), -1, 1)))
