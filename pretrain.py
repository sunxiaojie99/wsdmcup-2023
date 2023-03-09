# -*- encoding: utf-8 -*-
'''
@Time    :   2022/06/10 15:51:44
@Author  :   Chu Xiaokai
@Contact :   xiaokaichu@gmail.com
'''

# use this file to load the model from paddle, but do not publish this file
# python load_pretrain_model.py --emb_dim 768 --nlayer 3 --nhead 12 --dropout 0.1 --buffer_size 20 --eval_batch_size 20 --valid_click_path ./data/train_data/test.data.gz --save_step 5000 --init_parameters ./model3.pdparams --n_queries_for_each_gpu 10 --num_candidates 6
import time
import sys
import os
from dataloader import TrainDataset, TestDataset, mask_data
from Transformer4Ranking.model import TransformerModel, get_linear_schedule_with_warmup
import paddle
from paddle import nn
from paddle.io import DataLoader
import paddle.distributed as dist
from metrics import evaluate_all_metric
from args import config
import numpy as np
import random
import datetime

# control seed
# 生成随机数，以便固定后续随机数，方便复现代码
sys.path.append(os.getcwd())
random.seed(config.seed)
np.random.seed(config.seed)
paddle.set_device("gpu:0")
paddle.seed(config.seed)
print(config)


def save_dict(save_path, dict_need):
    import json
    with open(save_path, 'w', encoding='utf-8') as f_out:
        json.dump(dict_need, f_out, ensure_ascii=False, indent=2)


def format_time(time):
    elapsed_rounded = int(round((time)))
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


if not os.path.exists(config.output_name):  # 判断所在目录下是否有该文件名的文件夹
    os.makedirs(config.output_name)

valid_dict, test_dict = {}, {}
loss_dict = {}

vaild_annotate_dataset = TestDataset(
    config.valid_annotate_path, max_seq_len=config.max_seq_len, data_type='annotate')
vaild_annotate_loader = DataLoader(
    vaild_annotate_dataset, batch_size=config.eval_batch_size)

model = TransformerModel(
    ntoken=config.ntokens,
    hidden=config.emb_dim,
    nhead=config.nhead,
    nlayers=config.nlayers,
    dropout=config.dropout,
    mode='pretrain'
)
# load pretrained model
if config.init_parameters != "":
    print('load warm up model ', config.init_parameters)
    ptm = paddle.load(config.init_parameters)
    for k, v in model.state_dict().items():
        if k not in ptm:
            pass
            print("warning: not loading " + k)
        else:
            print("loading " + k)
            v.set_value(ptm[k])

# scheduler = get_linear_schedule_with_warmup(config.lr, config.warmup_steps,
#                                             config.max_steps)
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]
optimizer = paddle.optimizer.AdamW(
    learning_rate=config.lr,
    parameters=model.parameters(),
    weight_decay=config.weight_decay,
    apply_decay_param_fun=lambda x: x in decay_params,
    grad_clip=nn.ClipGradByNorm(clip_norm=0.5))
log_interval = config.log_interval
criterion = nn.BCEWithLogitsLoss()
# DataParallel
# initialize parallel environment

if paddle.device.cuda.device_count() >= config.n_gpus > 1:
    print("Let's use", config.n_gpus, "GPUs!")
    dist.init_parallel_env()
    model = paddle.DataParallel(model)

for epoch in range(config.pretrain_epoch):
    # train model
    model.train()  # turn on train mode
    log_interval = config.log_interval
    total_mlm_loss = 0.0
    valid_dict[epoch] = {}
    loss_dict[epoch] = {}
    best_eval_step = 0
    best_eval_score = None
    start_time, epoch_start_time = time.time(), time.time()

    idx = 0
    # load dataset
    train_dataset = TrainDataset(
        config.train_datadir, max_seq_len=config.max_seq_len, buffer_size=config.buffer_size)
    train_data_loader = DataLoader(
        train_dataset, batch_size=config.train_batch_size)
    for src_input, src_segment, src_padding_mask, click_label in train_data_loader:
        model.train()
        optimizer.clear_grad()
        masked_src_input, mask_label = mask_data(src_input)
        score, mlm_loss = model(
            src=masked_src_input,   # mask data
            src_segment=src_segment,
            src_padding_mask=src_padding_mask,
            mlm_label=mask_label,
        )
        mlm_loss = paddle.mean(mlm_loss)
        loss = mlm_loss
        loss.backward()
        optimizer.step()
        # scheduler.step()
        total_mlm_loss += mlm_loss.item()
        # log time
        if idx % log_interval == 0:
            # lr = scheduler.get_lr()
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_mlmloss = total_mlm_loss / log_interval
            print(
                f'{idx:5d}th step | '
                f'ms/batch {ms_per_batch:5.2f} | '
                f'mlm {cur_mlmloss:5.5f}')
            loss_dict[epoch][idx] = cur_mlmloss
            total_mlm_loss = 0
            start_time = time.time()

        # evaluate
        if idx % config.eval_step == 0:
            all_ndcg_list = []
            model.eval()

            # ------------   evaluate on annotated data -------------- #
            total_scores = []
            for src_input, src_segment, src_padding_mask, _ in vaild_annotate_loader:
                score = model(src=src_input, src_segment=src_segment,
                              src_padding_mask=src_padding_mask).cpu().detach().numpy().tolist()
                total_scores += score
            result_dict_ann = evaluate_all_metric(
                qid_list=vaild_annotate_dataset.total_qids,
                label_list=vaild_annotate_dataset.total_labels,
                score_list=total_scores,
                freq_list=vaild_annotate_dataset.total_freqs
            )
            valid_dict[epoch][idx] = {
                'dcg@10': result_dict_ann["all_dcg@10"],
                'high': result_dict_ann["high_dcg@10"],
                'mid': result_dict_ann["mid_dcg@10"],
                'low': result_dict_ann["low_dcg@10"],
                'pnr': result_dict_ann["pnr"]
            }
            print(
                f'{idx}th step valid annotate | '
                f'dcg@10: all {result_dict_ann["all_dcg@10"]:.5f} | '
                f'high {result_dict_ann["high_dcg@10"]:.5f} | '
                f'mid {result_dict_ann["mid_dcg@10"]:.5f} | '
                f'low {result_dict_ann["low_dcg@10"]:.5f} | '
                f'pnr {result_dict_ann["pnr"]:.5f}'
            )
            eval_score = result_dict_ann["all_dcg@10"]
            if best_eval_score is None or eval_score > best_eval_score:
                best_eval_score = eval_score
                best_eval_step = idx
                paddle.save(model.state_dict(),
                            config.output_name + '/best_model.model')
                print('new top test score {} at step-{}, saving weights'.format(
                    best_eval_score, best_eval_step))
                test_dict['best_step_{}'.format(
                    best_eval_step)] = best_eval_score
            save_dict(config.output_name + '/valid_dict.json', valid_dict)
            save_dict(config.output_name + '/loss.json', loss_dict)

        idx += 1
    paddle.save(model.state_dict(), config.output_name +
                '/pretrain_epoch{}_{:.5f}.model'.format(epoch, best_eval_score))
    print('epoch time:', format_time(time.time()-epoch_start_time))
