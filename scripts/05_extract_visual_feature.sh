#!/bin/bash
video_folder="$HOME/autodl-tmp/data/SumMe/videos"
output_folder="$HOME/autodl-tmp/data/SumMe/features/visual"
stride=15
batch_size=128

python process/05_extract_visual_feature.py \
    --video_folder $video_folder \
    --output_folder $output_folder \
    --stride $stride \
    --batch_size $batch_size


video_folder="$HOME/autodl-tmp/data/TVSum/videos"
output_folder="$HOME/autodl-tmp/data/TVSum/features/visual"
stride=15
batch_size=128

python process/05_extract_visual_feature.py \
    --video_folder $video_folder \
    --output_folder $output_folder \
    --stride $stride \
    --batch_size $batch_size
