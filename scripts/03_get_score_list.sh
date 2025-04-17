#!bash/bin
result_dir=$HOME/TFVSN/dataset/result/raw
dataset_dir=$HOME/autodl-tmp/data

python process/03_get_score_list.py \
    --dataset_dir $dataset_dir \
    --result_dir $result_dir
