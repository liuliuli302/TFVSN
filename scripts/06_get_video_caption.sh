#!/bin/bash

# 第一个数据集的处理
video_folder="$HOME/autodl-tmp/data/SumMe/videos"
output_folder="$HOME/autodl-tmp/data/SumMe/captions"

python process/06_get_video_caption.py \
    --video_folder $video_folder \
    --output_folder $output_folder

# 第二个数据集的处理
video_folder="$HOME/autodl-tmp/data/TVSum/videos"
output_folder="$HOME/autodl-tmp/data/TVSum/captions"

python process/06_get_video_caption.py \
    --video_folder $video_folder \
    --output_folder $output_folder
    