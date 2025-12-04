#!/bin/bash

CUDA="5"
NUM_GPU=1

CUDA_VISIBLE_DEVICES=${CUDA} accelerate launch --num_processes ${NUM_GPU} test/test.py \
    --config run_configs/val/val_satext_lv3_dit4sr_testr.yaml

