#! /bin/bash

srun --job-name=warp \
    --gres=gpu:1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --kill-on-bad-exit=1 \
    --time=24:00:00 \
    python $NFS/code/sinf/sinf/seg/warp.py -j=$1