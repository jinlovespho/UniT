#!/bin/bash

CUDA="0,1,2,3"
NUM_GPU=4

CUDA_VISIBLE_DEVICES=${CUDA} accelerate launch  --num_processes ${NUM_GPU} train/train_dit4sr.py \
                                                --config run_configs/train/JIHYE_train_stage3_dit4sr_testr.yaml

