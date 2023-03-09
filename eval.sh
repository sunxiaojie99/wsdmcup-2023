layer=3

feature_type='JM_title'

# 添加fetaure：combine=1
combine=1  # 是否在transformer 用feature做输入

# 添加投影：projection=1, rank_feature_size=869
# 不添加投影: projection=0, rank_feature_size=792
projection=1
rank_feature_size=834

# dla 添加矫正 change_label='max' or 'delta', delta=xx, temperature=xx
change_label='max'  # 'no'
vote=1

# pairwise: mode='pair'
# dla: mode='list'
# dla-替换最后一次点击后的click=0为random neg: mode='replace_list'
mode='list'

method_name='DLA'  # NaiveAlgorithm DLA
# mode='pair' 的时候改一下 feature_pairs


# --test_data_path "/ossfs/workspace/wsdm_cup/data/annotate_data/wsdm_test_2_all.txt"  \
# --test_feature_path '/ossfs/workspace/wsdm_cup/data/new_train/features/wsdm_test_2_all.txt' \

# --test_data_path "/ossfs/workspace/wsdm_cup/data/new_train/finetune_dev.txt"  \
# --test_feature_path '/ossfs/workspace/wsdm_cup/data/new_train/features/finetune_dev.txt' \

CUDA_VISIBLE_DEVICES=0 python -u submit_unbiased.py \
--emb_dim 768 --nlayer $layer --nhead 12 \
--init_parameters '/ossfs/workspace/wsdm_cup/output/3l_dla_combine834_neg10_list_vote1_replacequerylen/best_model.model' \
--test_data_path "/ossfs/workspace/wsdm_cup/data/annotate_data/wsdm_test_2_all.txt"  \
--test_feature_path '/ossfs/workspace/wsdm_cup/data/new_train/features/wsdm_test_2_all.txt' \
--eval_batch_size 2000 \
--method_name $method_name  \
--n_gpus 1 \
--combine $combine \
--change_label $change_label \
--feature_type $feature_type \
--projection $projection \
--rank_feature_size $rank_feature_size \
--mode $mode \
--vote $vote \
--result_path '/ossfs/workspace/wsdm_cup/output/3l_dla_combine834_neg10_list_vote1_replacequerylen/best_test2.csv'
