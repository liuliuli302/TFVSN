#!bin/bash
# Extract SumMe
input_root=$HOME/autodl-tmp/data/SumMe/videos
save_root=$HOME/autodl-tmp/data/SumMe/features/visual
stride=15
batch_size=128

python process/06_extract_visual_feature.py \
    --input_root $input_root \
    --save_root $save_root \
    --stride $stride \
    --batch_size $batch_size


# Extract TVSum
input_root=$HOME/autodl-tmp/data/TVSum/videos
save_root=$HOME/autodl-tmp/data/TVSum/features/visual
stride=15
batch_size=128

python process/06_extract_visual_feature.py \
    --input_root $input_root \
    --save_root $save_root \
    --stride $stride \
    --batch_size $batch_size