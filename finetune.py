# -*- encoding: utf-8 -*-
'''
@Time    :   2022/06/12 14:49:28
@Author  :   Chu Xiaokai 
@Contact :   xiaokaichu@gmail.com
'''
import numpy as np
import warnings
import sys
from metrics import evaluate_all_metric
from Transformer4Ranking.model import *
from paddle.io import DataLoader
from dataloader import TestDataset
from args import config
import paddle
import random
import paddle.distributed as dist
import time
import datetime
import paddle.nn.functional as F

paddle.disable_signal_handler()

random.seed(config.seed+1)
random.seed(config.seed)
np.random.seed(config.seed)

# 1. 多卡改动1
paddle.set_device("gpu:0")
# dist.init_parallel_env()

paddle.seed(config.seed)
print(config)
exp_settings = config.exp_settings


def softmax_cross_entropy_with_logits(logits, labels):
    """Computes softmax cross entropy between logits and labels.

    Args:
        output: A tensor with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
        labels: A tensor of the same shape as `output`. A value >= 1 means a
        relevant example.
    Returns:
        A single value tensor containing the loss.
    """
    loss = paddle.sum(- labels * F.log_softmax(logits, axis=-1), axis=-1)
    return loss


def save_dict(save_path, dict_need):
    import json
    with open(save_path, 'w', encoding='utf-8') as f_out:
        json.dump(dict_need, f_out, ensure_ascii=False, indent=2)


combine = config.combine
projection = config.projection
rank_feature_size = config.rank_feature_size

model = TransformerModel(
    ntoken=config.ntokens,
    hidden=config.emb_dim,
    nhead=config.nhead,
    nlayers=config.nlayers,
    dropout=config.dropout,
    mode='task1_finetune',
    combine=combine,
    projection=projection,
    rank_feature_size=rank_feature_size,
)

# load pretrained model
if config.init_parameters != "":
    print('load warm up model ', config.init_parameters)
    ptm = paddle.load(config.init_parameters)
    for k, v in model.state_dict().items():
        if not k in ptm:
            pass
            print("warning: not loading " + k)
        else:
            print("loading " + k)
            v.set_value(ptm[k])

# 优化器设置
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
    grad_clip=nn.ClipGradByNorm(clip_norm=0.5)
)

finetune_type = config.finetune_type
if finetune_type == 'finetune_point':
    vaild_annotate_dataset = TestDataset(
        config.valid_annotate_path, max_seq_len=config.max_seq_len, data_type='finetune', config=config)
    vaild_annotate_loader = DataLoader(
        vaild_annotate_dataset, batch_size=config.eval_batch_size)

test_annotate_dataset = TestDataset(
    config.test_annotate_path, max_seq_len=config.max_seq_len, data_type='annotate',
    config=config, load_feature=combine, feature_path=config.test_feature_path)
test_annotate_loader = DataLoader(
    test_annotate_dataset, batch_size=config.eval_batch_size)

idx = 0
t0 = time.time()
best_eval_step = 0
best_eval_score = None
result_dict = {}
loss_dict = {}
for epoch_idx in range(config.finetune_epoch):
    print('=====epoch:{}====='.format(epoch_idx))
    if finetune_type == 'finetune_pair' or finetune_type == 'both':
        epoch_vaild_annotate_dataset = TestDataset(
            config.valid_annotate_path, max_seq_len=config.max_seq_len, data_type='finetune_pair', config=config,
            feature_path=config.train_feature_path, load_feature=combine)
        vaild_annotate_loader = DataLoader(
            epoch_vaild_annotate_dataset, batch_size=config.eval_batch_size, drop_last=True)
    elif finetune_type == 'finetune_list':
        print('=====finetune_list')
        epoch_vaild_annotate_dataset = TestDataset(
            config.valid_annotate_path, max_seq_len=config.max_seq_len, data_type='finetune_list', config=config,
            feature_path=config.train_feature_path, load_feature=combine)
        vaild_annotate_loader = DataLoader(
            epoch_vaild_annotate_dataset, batch_size=config.eval_batch_size, drop_last=True)

    for valid_data_batch in vaild_annotate_loader:
        model.train()
        optimizer.clear_grad()
        if (finetune_type == 'finetune_pair' or finetune_type == 'both') and combine == 1:
            src_input, src_segment, src_padding_mask, label = valid_data_batch[0]
            features = valid_data_batch[1]
            score = model(
                src=src_input,
                src_segment=src_segment,
                src_padding_mask=src_padding_mask,
                features=features
            )
        else:
            src_input, src_segment, src_padding_mask, label = valid_data_batch
            score = model(
                src=src_input,
                src_segment=src_segment,
                src_padding_mask=src_padding_mask,
            )
        if finetune_type == 'finetune_point':
            criterion = nn.BCEWithLogitsLoss()
            ctr_loss = criterion(score, paddle.to_tensor(
                label, dtype=paddle.float32))

        elif finetune_type == 'finetune_pair':
            loss_fct = nn.CrossEntropyLoss()
            subset_num = config.negative_num + 1
            # [batch_q_nums, negative_num + 1]
            postive_logits = score.reshape(
                [config.eval_batch_size//subset_num, subset_num])
            pairwise_labels = paddle.zeros(
                [config.eval_batch_size//subset_num], dtype='int64')
            ctr_loss = loss_fct(postive_logits / config.tem, pairwise_labels)
        elif finetune_type == 'both':
            loss_fct = nn.CrossEntropyLoss()
            subset_num = config.negative_num + 1
            # [batch_q_nums, negative_num + 1]
            postive_logits = score.reshape(
                [config.eval_batch_size//subset_num, subset_num])
            pairwise_labels = paddle.zeros(
                [config.eval_batch_size//subset_num], dtype='int64')
            pair_loss = loss_fct(postive_logits / config.tem, pairwise_labels)

            criterion = nn.BCEWithLogitsLoss()
            point_loss = criterion(score, paddle.to_tensor(
                label, dtype=paddle.float32))
            ctr_loss = pair_loss + point_loss
        elif finetune_type == 'finetune_list':
            subset_num = config.negative_num + config.pos_num
            logits = score.reshape(
                [config.eval_batch_size//subset_num, subset_num])
            label = paddle.to_tensor(label, dtype=paddle.float32)
            label = label.reshape([config.eval_batch_size//subset_num, subset_num])

            batch_loss = softmax_cross_entropy_with_logits(logits, label)
            ctr_loss = paddle.mean(batch_loss)

        ctr_loss.backward()
        optimizer.step()
        # scheduler.step()

        if idx % config.log_interval == 0:
            print(f'{idx:5d}th step | loss {ctr_loss.item():5.6f}')
            loss_dict[idx] = ctr_loss.item()

        if idx % config.eval_step == 0:
            model.eval()
            # ------------   evaluate on annotated data -------------- #
            total_scores = []
            for test_data_batch in test_annotate_loader:
                if combine == 1:
                    src_input, src_segment, src_padding_mask, label, features = test_data_batch
                    score = model(
                        src=src_input,
                        src_segment=src_segment,
                        src_padding_mask=src_padding_mask,
                        features=features
                    )
                else:
                    src_input, src_segment, src_padding_mask, label = test_data_batch
                    score = model(
                        src=src_input,
                        src_segment=src_segment,
                        src_padding_mask=src_padding_mask,
                    )
                score = score.cpu().detach().numpy().tolist()
                total_scores += score

            result_dict_ann = evaluate_all_metric(
                qid_list=test_annotate_dataset.total_qids,
                label_list=test_annotate_dataset.total_labels,
                score_list=total_scores,
                freq_list=test_annotate_dataset.total_freqs
            )
            result_dict[idx] = {
                'dcg@10': result_dict_ann["all_dcg@10"],
                'high': result_dict_ann["high_dcg@10"],
                'mid': result_dict_ann["mid_dcg@10"],
                'low': result_dict_ann["low_dcg@10"],
                'pnr': result_dict_ann["pnr"],
            }
            print(
                f'{idx}th step valid annotate | '
                f'dcg@10: all {result_dict_ann["all_dcg@10"]:.6f} | '
                f'high {result_dict_ann["high_dcg@10"]:.6f} | '
                f'mid {result_dict_ann["mid_dcg@10"]:.6f} | '
                f'low {result_dict_ann["low_dcg@10"]:.6f} | '
                f'pnr {result_dict_ann["pnr"]:.6f}'
            )
            if idx % config.save_step == 0 and idx > 0:
                paddle.save(model.state_dict(),
                            config.output_name + '/save_steps{}_{:.5f}.model'.format(
                                idx, result_dict_ann['pnr'])
                            )
            eval_score = result_dict_ann["all_dcg@10"]
            if best_eval_score is None or eval_score > best_eval_score:
                best_eval_score = eval_score
                best_eval_step = idx
                paddle.save(model.state_dict(),
                            config.output_name + '/best_model.model')
                print('new top validation score {} at step-{}, saving weights'.format(
                    best_eval_score, best_eval_step))
                result_dict['best_step_{}'.format(
                    best_eval_step)] = best_eval_score
            save_dict(config.output_name + '/result_dict.json', result_dict)
            save_dict(config.output_name + '/loss.json', loss_dict)
        idx += 1
t1 = time.time()
training_time = t1 - t0


def format_time(time):
    elapsed_rounded = int(round((time)))
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


training_time = format_time(training_time)
print('training_time', training_time)

save_dict(config.output_name + '/result_dict.json', result_dict)
save_dict(config.output_name + '/loss.json', loss_dict)
