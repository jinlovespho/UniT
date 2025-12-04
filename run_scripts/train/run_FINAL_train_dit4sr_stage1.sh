#!/bin/bash

CUDA="6"
NUM_GPU=1

CUDA_VISIBLE_DEVICES=${CUDA} accelerate launch  --num_processes ${NUM_GPU} train/train_dit4sr.py \
                                                --config run_configs/train/FINAL_train_dit4sr_stage1.yaml
