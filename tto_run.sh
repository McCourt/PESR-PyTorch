dataset='Set5'
model='EnhanceNet-E'
scale=4

python3 -W ignore ./tto.py \
  --model-name=${model} \
  --dataset=${dataset} \
  --scale=${scale}
