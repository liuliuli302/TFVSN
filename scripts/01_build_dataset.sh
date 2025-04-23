#!/bin/bash
dataset_dir=$HOME/autodl-tmp/data
dataset_save_dir=$HOME/TFVSN/dataset
clip_length=5

python process/01_build_dataset.py \
    --dataset_dir $dataset_dir \
    --dataset_save_dir $dataset_save_dir \
    --clip_length $clip_length
