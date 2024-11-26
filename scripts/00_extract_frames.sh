#!/bin/bash
dataset_dir="$HOME/workspace/TFVSN/data/TVSum"

# Set paths
videos_dir="${dataset_dir}/videos"
frames_dir="${dataset_dir}/frames"
annotations_file="${dataset_dir}/annotations/test.txt"

python src/util/extract_frames.py \
    --videos_dir "$videos_dir" \
    --frames_dir "$frames_dir" \
    --annotations_file "$annotations_file"
