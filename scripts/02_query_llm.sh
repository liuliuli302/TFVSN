#!bash/bin
dataset_save_dir=$HOME/TFVSN/dataset
result_dir=$HOME/TFVSN/dataset/result/raw

python process/02_query_llm.py \
    --dataset_save_dir $dataset_save_dir \
    --result_dir $result_dir
