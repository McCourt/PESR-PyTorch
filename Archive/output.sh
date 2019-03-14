#!/usr/bin/env bash
# python3 -W ignore output.py \
#   --model-name='srresnet' \
#   --lr-dir='/home/mccourt/superresolution/demo/valid_lr' \
#   --hr-dir='/home/mccourt/superresolution/demo/valid_hr' \
#   --output-dir='/home/mccourt/superresolution/outputs' \
#   --gpu-device='cuda' \
#   --old-loc='None' \
#   --ckpt-dir='/home/mccourt/superresolution/checkpoints' \
#   --upsample='false'
folder='Set5'
model='srresnet'
python3 -W ignore output.py \
  --model-name=$model \
  --lr-dir="/home/mccourt/SR_testing_datasets/${folder}_x4_ds" \
  --hr-dir="/home/mccourt/SR_testing_datasets/$folder" \
  --output-dir="/home/mccourt/superresolution/outputs/$folder" \
  --gpu-device='cuda' \
  --old-loc='None' \
  --ckpt-dir='/home/mccourt/superresolution/checkpoints' \
  --upsample='false'
