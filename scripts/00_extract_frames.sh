#!/bin/bash
dataset_dir="$HOME/autodl-tmp/data/SumMe"

# Set paths
videos_dir="${dataset_dir}/videos"
frames_dir="${dataset_dir}/frames"
annotations_file="${dataset_dir}/annotations/test.txt"

python processing/00_extract_frames.py \
    --videos_dir "$videos_dir" \
    --frames_dir "$frames_dir" \
    --annotations_file "$annotations_file"
