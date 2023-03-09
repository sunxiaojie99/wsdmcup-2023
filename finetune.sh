# 如果是pairwise，那么需要eval_batch_size%(negative_num+1)==0
output_name='base2-pair-bs352-lr2e-7-neg10'
finetune_epoch=20
eval_batch_size=352  # 对于pair，需要时neg+1的倍数；对于list，需要是neg+pos的倍数
negative_num=10  # 对于list，就是pos和doc各取几个
pos_num=10
lr=2e-7
tem=1

finetune_type='finetune_pair'  # finetune_list or finetune_pair or finetune_point

combine=0

CUDA_VISIBLE_DEVICES=1 python -u finetune.py \
    --emb_dim 768 \
    --nlayer 12 --nhead 12 \
    --eval_batch_size $eval_batch_size \
    --dropout 0.1 --n_gpus 1 \
    --init_parameters 'output/pair-bs288-lr2e-7-neg5/best_model.model' \
    --valid_annotate_path data/new_train/finetine_train.txt \
    --test_annotate_path data/new_train/finetune_dev.txt \
    --finetune_epoch $finetune_epoch \
    --negative_num $negative_num \
    --strategy 'd' \
    --lr $lr \
    --finetune_type $finetune_type \
    --output_name $output_name \
    --eval_step 500 \
    --tem $tem \
    --combine $combine \
    --pos_num $pos_num \
    > $output_name.log 2>&1