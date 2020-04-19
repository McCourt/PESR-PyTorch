#!/bin/csh -f
#SBATCH --output=SISR.out
#SBATCH --error=SISR.err
#SBATCH -p compsci-gpu --gres=gpu:4 --exclude=gpu-compute[1-3]

python3 -W ignore main.py --mode='train'
