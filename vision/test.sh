#!/bin/bash
MODEL=shufflenet_v2
WORKERS=16
CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --model $MODEL \
    --epochs 300 \
    --batch-size 64 \
    --lr 0.045 \
    --wd 0.00004 \
    --lr-step-size 1 \
    --lr-gamma 0.98 \
    --output-dir=./checkpoint/$MODEL \
    --print-freq=100 \
    --workers=$WORKERS \
    --test-time \
    --data-path $HOME/datasets/imagenet/ \
    | & tee -a ./checkpoint/$MODEL/log
