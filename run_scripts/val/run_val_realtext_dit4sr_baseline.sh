#!/bin/bash

CUDA="2"
NUM_GPU=1

CUDA_VISIBLE_DEVICES=${CUDA} accelerate launch --num_processes ${NUM_GPU} test/test.py \
    --config run_configs/val/val_realtext_dit4sr_baseline.yaml

# if gpu id doesnt work delete the default config in the following location.
# /home/cvlab20/.cache/huggingface/accelerate