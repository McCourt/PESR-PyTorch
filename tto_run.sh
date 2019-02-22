dataset='Urban100'
model='EnhanceNet-E'
method='convolution_down_sample'
scale=4
path="/usr/xtmp/superresoluter/superresolution"

python3 -W ignore ${path}/tto.py \
  --model-name=${model} \
  --dataset=${dataset} \
  --method=${method} \
  --scale=${scale}
