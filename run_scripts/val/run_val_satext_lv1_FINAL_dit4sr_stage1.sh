#!/bin/bash

CUDA="6"
NUM_GPU=1

CUDA_VISIBLE_DEVICES=${CUDA} accelerate launch --num_processes ${NUM_GPU} test/test.py \
                --config run_configs/val/val_satext_lv1_FINAL_dit4sr_stage1.yaml

