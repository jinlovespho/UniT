#!/bin/bash

CUDA="4"
NUM_GPU=1

CUDA_VISIBLE_DEVICES=${CUDA} accelerate launch --num_processes ${NUM_GPU} test/test.py \
                --config run_configs/val/val_satext_test_FINAL_dit4sr_stage2.yaml

