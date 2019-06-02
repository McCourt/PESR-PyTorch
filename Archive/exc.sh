#!/usr/bin/env bash
folder='Set14'
python3 -W ignore /home/mccourt/superresolution/trainer.py \
    --mode='output'
    --model-name='srresnet' \
    --lr-dir="/home/mccourt/SR_testing_datasets/${folder}_x4_ds" \
    --hr-dir="/home/mccourt/SR_testing_datasets/${folder}" \
    --output-dir="/home/mccourt/superresolution/outputs/${folder}" \
    --gpu-device='cuda' \
    --ckpt-dir='/home/mccourt/superresolution/checkpoints'
