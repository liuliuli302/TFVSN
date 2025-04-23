visual_feature_dir=$HOME/autodl-tmp/data/SumMe/features/visual
text_feature_dir=$HOME/autodl-tmp/data/SumMe/features/text
segment_num=4
out_put_dir=$HOME/TFVSN/dataset/result/similarity_scores/SumMe


python process/08_calc_semantic_similarity.py \
    --visual_feature_dir $visual_feature_dir \
    --text_feature_dir $text_feature_dir \
    --segment_num $segment_num \
    --out_put_dir $out_put_dir


visual_feature_dir=$HOME/autodl-tmp/data/TVSum/features/visual
text_feature_dir=$HOME/autodl-tmp/data/TVSum/features/text
segment_num=4
out_put_dir=$HOME/TFVSN/dataset/result/similarity_scores/TVSum


python process/08_calc_semantic_similarity.py \
    --visual_feature_dir $visual_feature_dir \
    --text_feature_dir $text_feature_dir \
    --segment_num $segment_num \
    --out_put_dir $out_put_dir
