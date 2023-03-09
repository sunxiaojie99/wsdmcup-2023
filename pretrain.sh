output_name='pretrain_mlm_12l_retrain'
layer=12
epoch=10
CUDA_VISIBLE_DEVICES=0 python -u pretrain.py \
  --init_parameters '/ossfs/workspace/wsdm_cup/output/pretrain_mlm_12l/pretrain_epoch0_7.02422.model' \
  --valid_annotate_path /ossfs/workspace/wsdm_cup/data/new_train/finetune_dev.txt \
  --emb_dim 768 \
  --nlayer $layer \
  --dropout 0.1 \
  --buffer_size 20000 \
  --n_gpus 1 \
  --log_interval 50 \
  --eval_step 1000 \
  --pretrain_epoch $epoch \
  --n_queries_for_each_gpu 10 \
  --num_candidates 10 \
  --eval_batch_size 200 \
  --lr 2e-6 \
  --output_name 'output/'$output_name > $output_name.log 2>&1