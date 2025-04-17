#!/bin/bash
dataset_dir="$HOME/autodl-tmp/data"

tvsum_videos_dir="${dataset_dir}/TVSum/videos"
tvsum_frames_dir="${dataset_dir}/TVSum/frames"
tvsum_annotations_file="${dataset_dir}/TVSum/annotations/test.txt"

summe_videos_dir="${dataset_dir}/SumMe/videos"
summe_frames_dir="${dataset_dir}/SumMe/frames"
summe_annotations_file="${dataset_dir}/SumMe/annotations/test.txt"

python processing/00_extract_frames.py \
    --videos_dir "$tvsum_videos_dir" \
    --frames_dir "$tvsum_frames_dir" \
    --annotations_file "$tvsum_annotations_file"

python processing/00_extract_frames.py \
    --videos_dir "$summe_videos_dir" \
    --frames_dir "$summe_frames_dir" \
    --annotations_file "$summe_annotations_file"
