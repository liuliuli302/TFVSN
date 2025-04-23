#!/bin/bash
text_folder="$HOME/autodl-tmp/data/SumMe/captions"
output_folder="$HOME/autodl-tmp/data/SumMe/features/text"

python process/07_extract_text_feature.py \
    --text_folder $text_folder \
    --output_folder $output_folder \


text_folder="$HOME/autodl-tmp/data/TVSum/captions"
output_folder="$HOME/autodl-tmp/data/TVSum/features/text"

python process/07_extract_text_feature.py \
    --text_folder $text_folder \
    --output_folder $output_folder \
