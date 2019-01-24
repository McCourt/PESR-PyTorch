dataset='Urban100'
model='EnhanceNet-E'
method='trained_conv'
scale=4
path="/usr/xtmp/superresoluter/superresolution"

python3 -W ignore ${path}/tester_trained_ds.py \
  --model-name=${model} \
  --dataset=${dataset} \
  --method=${method} \
  --scale=${scale}
