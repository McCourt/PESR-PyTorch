dataset='Set5'
model='EnhanceNet-E'
scale=4

source ../sr/bin/activate.csh
python3 -W ignore ${path}/tto.py \
  --model-name=${model} \
  --dataset=${dataset} \
  --scale=${scale}
