#!bash/bin
score_file=$HOME/TFVSN/dataset/result/scores/raw_llm_out_scores.json
dataset_dir=$HOME/autodl-tmp/data

python process/04_eval_llm_output.py \
    --dataset_dir $dataset_dir \
    --score_file $score_file