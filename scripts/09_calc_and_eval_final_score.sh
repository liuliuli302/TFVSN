#!/bin/bash
# 设置路径变量（请根据你的实际路径修改）
LLM_SCORE_PATH="/root/TFVSN/dataset/result/scores/raw_llm_out_scores.json"
SIM_SCORE_DIR="/root/TFVSN/dataset/result/similarity_scores"
DATA_DIR="/root/autodl-tmp/data"
OUTPUT_DIR="/root/TFVSN/dataset/result/f1score"

# 运行 Python 脚本
python process/09_calc_and_eval_final_score.py \
    --llm_score "$LLM_SCORE_PATH" \
    --sim_score_dir "$SIM_SCORE_DIR" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"
