output_name='3l_dla_combine834_neg10_list_vote1_replacequerylen'


neg_num=10 # 负样本数量
task1_epoch=1
freq_k=1.5
freq_b=0.75 


# 添加fetaure：train_load_feature=1，combine=1
combine=1  # 是否在transformer 用feature做输入
train_load_feature=1  # 是否读取 train feature

# 添加投影：projection=1, rank_feature_size=870
# 不添加投影: projection=0, rank_feature_size=793
projection=1
rank_feature_size=834  # 768+66

# dla 添加矫正 change_label='max' or 'delta', delta=xx, temperature=xx
change_label='max'  # 'no'
temp=0.1
delta=2.0
feature_type='JM_title'
vote=1  # 是否使用vote得到的feature作为label，指定后，feature_type失效

# pairwise: mode='pair'
# dla: mode='list'
# dla-替换最后一次点击后的click=0为random neg: mode='replace_list'
mode='list'

method_name='DLA'  # NaiveAlgorithm DLA
# mode='pair' 的时候去args里改一下 feature_pairs


CUDA_VISIBLE_DEVICES=1 python -u ./unbiased_learning.py \
    --init_parameters /ossfs/workspace/wsdm_cup/data/pre_model/baidu_ultr_3l_12h_768e.model \
    --emb_dim 768 \
    --nlayer 3 \
    --nhead 12 \
    --dropout 0.1 \
    --train_datadir /ossfs/workspace/wsdm_cup/data/train_data/ \
    --train_feature_path /ossfs/workspace/wsdm_cup/data/new_train/features/norm_part-00001.txt \
    --valid_data_path '' \
    --valid_feature_path '' \
    --test_data_path /ossfs/workspace/wsdm_cup/data/new_train/finetune_train.txt \
    --test_feature_path /ossfs/workspace/wsdm_cup/data/new_train/features/finetune_train.txt \
    --num_candidates 10 \
    --method_name $method_name  \
    --buffer_size 20000 \
    --n_gpus 1 \
    --negative_num $neg_num \
    --mean_std_file /ossfs/workspace/wsdm_cup/data/new_train/features/mean_std.txt \
    --feature_type $feature_type \
    --temperature $temp \
    --output_name 'output/'$output_name \
    --eval_step 500 \
    --save_step 5000 \
    --n_queries_for_each_gpu 11 \
    --change_label $change_label \
    --lr 2e-6 \
    --delta $delta \
    --projection $projection \
    --rank_feature_size $rank_feature_size \
    --train_load_feature $train_load_feature \
    --combine $combine \
    --mode $mode \
    --task1_epoch $task1_epoch \
    --vote $vote \
    --freq_path "/ossfs/workspace/wsdm_cup/data/new_train/query_frequency.txt" \
    --add_freq "False" \
    --freq_k $freq_k \
    --freq_b $freq_b \
    > $output_name.log 2>&1

# --add_freq "True"