#!/bin/bash

CUDA="5,6,7"
NUM_GPU=3

CUDA_VISIBLE_DEVICES=${CUDA} accelerate launch  --num_processes ${NUM_GPU} train/train_dit4sr.py \
                                                --config run_configs/train/train_stage1_dit4sr.yaml
