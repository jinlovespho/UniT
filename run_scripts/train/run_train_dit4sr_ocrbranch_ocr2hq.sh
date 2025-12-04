#!/bin/bash

CUDA="0"
NUM_GPU=1

CUDA_VISIBLE_DEVICES=${CUDA} accelerate launch  --num_processes ${NUM_GPU} train/train_dit4sr.py \
                                                --config run_configs/train/train_dit4sr_ocrbranch_ocr2hq.yaml
