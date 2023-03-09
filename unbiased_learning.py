# -*- encoding: utf-8 -*-
from baseline_model.utils.sys_tools import find_class
import paddle
import numpy as np
import warnings
import sys
from metrics import *
from Transformer4Ranking.model import *
from paddle.io import DataLoader
from dataloader import *
from args import config
import random
import time
import os
import matplotlib.pyplot as plt

random.seed(config.seed+1)
random.seed(config.seed)
np.random.seed(config.seed)

paddle.set_device("gpu:0")
paddle.seed(config.seed)

print(config)
exp_settings = config.exp_settings
'''change_label控制是否修正label<=>只在训练时需要
   combine控制大模型是否加feature过DNN=>控制eval时是否读feature
   train_load_feature控制train时是否读features文件
'''
combine = config.combine
change_label = config.change_label
projection = config.projection
train_load_feature = config.train_load_feature
train_mode = config.mode


def save_dict(save_path, dict_need):
    import json
    with open(save_path, 'w', encoding='utf-8') as f_out:
        json.dump(dict_need, f_out, ensure_ascii=False, indent=2)


token_encoder = TransformerModel(
    ntoken=config.ntokens,
    hidden=config.emb_dim,
    nhead=config.nhead,
    nlayers=config.nlayers,
    dropout=config.dropout,
    mode='task1_finetune',
    combine=combine,
    projection=projection,
    rank_feature_size=config.exp_settings['rank_feature_size'],
    add_freq=config.add_freq
)
# load pretrained model
if config.init_parameters != "":
    print('load warm up model ', config.init_parameters)
    ptm = paddle.load(config.init_parameters)
    for k, v in token_encoder.state_dict().items():
        if not k in ptm:
            pass
            print("warning: not loading " + k)
        else:
            print("loading " + k)
            v.set_value(ptm[k])

# models_dir
if not os.path.exists('models/'):  # 判断所在目录下是否有该文件名的文件夹
    os.mkdir('models/')
if not os.path.exists(config.output_name):  # 判断所在目录下是否有该文件名的文件夹
    os.makedirs(config.output_name)


# load pretrained model=>baseline_model中load

method_str = exp_settings['method_name']
if method_str not in ['IPWrank', 'DLA', 'RegressionEM', 'PairDebias', 'NaiveAlgorithm']:
    print("please choose a method in 'IPWrank', 'DLA','RegressionEM','PairDebias','NaiveAlgorithm'")
    sys.exit()
model = find_class('baseline_model.learning_algorithm.' +
                   method_str)(exp_settings=exp_settings, encoder_model=token_encoder)

if config.valid_data_path != '':
    vaild_annotate_dataset = TestDataset(config.valid_data_path, max_seq_len=config.max_seq_len,
                                         data_type='task1_annotate', load_feature=combine, feature_path=config.valid_feature_path)
    vaild_annotate_loader = DataLoader(
        vaild_annotate_dataset, batch_size=config.eval_batch_size)
test_annotate_dataset = TestDataset(config.test_data_path, max_seq_len=config.max_seq_len,
                                    data_type='task1_annotate', load_feature=combine, feature_path=config.test_feature_path)
test_annotate_loader = DataLoader(
    test_annotate_dataset, batch_size=config.eval_batch_size)

idx = -1
start = time.time()

# loss DCG@10
valid_dict, test_dict = {}, {}
loss_dict = {}
best_eval_step = 0
best_eval_score = None

total_scores = []
for epoch in range(config.task1_epoch):
    train_dataset = Train_listDataset(config.train_datadir, max_seq_len=config.max_seq_len,
                                      buffer_size=config.buffer_size, load_feature=train_load_feature,
                                      num_candidates=config.num_candidates,
                                      negative_num=config.negative_num,
                                      query_num=config.n_gpus * config.n_queries_for_each_gpu,
                                      train_feature_path=config.train_feature_path, mode=train_mode, feature_pairs=config.feature_pairs,
                                    train_freq_path=config.freq_path)
    train_data_loader = DataLoader(
        train_dataset, batch_size=config.train_batch_size)
    for train_batch in train_data_loader:
        idx += 1
        loss = model.train(build_feed_dict(train_batch, load_feature=train_load_feature,
                                           size=config.num_candidates+config.negative_num, mode=train_mode))
        if idx % config.log_interval == 0:
            print(f'{idx:5d}th step | loss {loss:5.6f}')
            loss_dict[idx] = loss

        if idx % config.eval_step == 0:
            # ------------   evaluate on annotated data -------------- #

            # valid
            if config.valid_data_path != '':
                total_scores = []
                with paddle.no_grad():
                    for test_data_batch in vaild_annotate_loader:
                        feed_input = build_feed_dict(
                            test_data_batch, combine)
                        score = model.get_scores(feed_input)
                        score = score.cpu().detach().numpy().tolist()
                        total_scores += score

                result_dict_ann = evaluate_all_metric(
                    qid_list=vaild_annotate_dataset.total_qids,
                    label_list=vaild_annotate_dataset.total_labels,
                    score_list=total_scores,
                    freq_list=vaild_annotate_dataset.total_freqs
                )
                valid_dict[idx] = {
                    'dcg@10': result_dict_ann["all_dcg@10"],
                    'high': result_dict_ann["high_dcg@10"],
                    'mid': result_dict_ann["mid_dcg@10"],
                    'low': result_dict_ann["low_dcg@10"],
                    'pnr': result_dict_ann["pnr"],
                }
                print(
                    f'{idx}th step valid annotate | '
                    f'@10 dcg: all {result_dict_ann["all_dcg@10"]:.6f} | '
                    f'high {result_dict_ann["high_dcg@10"]:.6f} | '
                    f'mid {result_dict_ann["mid_dcg@10"]:.6f} | '
                    f'low {result_dict_ann["low_dcg@10"]:.6f} | '
                    f'err {result_dict_ann["all_err@10"]:.6f} | '
                    f'pnr {result_dict_ann["pnr"]:.6f}'
                )

            # test annotate

            total_scores = []
            with paddle.no_grad():
                for test_data_batch in test_annotate_loader:
                    feed_input = build_feed_dict(test_data_batch, combine)
                    score = model.get_scores(feed_input)
                    score = score.cpu().detach().numpy().tolist()
                    total_scores += score

            result_dict_test = evaluate_all_metric(
                qid_list=test_annotate_dataset.total_qids,
                label_list=test_annotate_dataset.total_labels,
                score_list=total_scores,
                freq_list=test_annotate_dataset.total_freqs
            )
            test_dict[idx] = {
                'dcg@10': result_dict_test["all_dcg@10"],
                'high': result_dict_test["high_dcg@10"],
                'mid': result_dict_test["mid_dcg@10"],
                'low': result_dict_test["low_dcg@10"],
                'pnr': result_dict_test["pnr"],
            }
            print(
                f'{idx}th step test annotate | '
                f'@10 dcg: all {result_dict_test["all_dcg@10"]:.6f} | '
                f'high {result_dict_test["high_dcg@10"]:.6f} | '
                f'mid {result_dict_test["mid_dcg@10"]:.6f} | '
                f'low {result_dict_test["low_dcg@10"]:.6f} | '
                f'err {result_dict_test["all_err@10"]:.6f} | '
                f'pnr {result_dict_test["pnr"]:.6f}'
            )
            eval_score = result_dict_test["all_dcg@10"]
            if best_eval_score is None or eval_score > best_eval_score:
                best_eval_score = eval_score
                best_eval_step = idx
                paddle.save(model.state_dict(),
                            config.output_name + '/best_model.model')
                print('new top test score {} at step-{}, saving weights'.format(
                    best_eval_score, best_eval_step))
                test_dict['best_step_{}'.format(
                    best_eval_step)] = best_eval_score
            save_dict(config.output_name + '/test_dict.json', test_dict)
            save_dict(config.output_name + '/valid_dict.json', valid_dict)
            save_dict(config.output_name + '/loss.json', loss_dict)

        if idx % config.save_step == 0 and idx > 0:
            paddle.save(model.state_dict(),
                        config.output_name + '/{}_{}_{}_{}_{:.5f}.model'.format(
                            exp_settings['method_name'], change_label, combine, idx /
                config.save_step,
                            result_dict_test['all_dcg@10'])
                        )

    # ------------   evaluate on annotated data -------------- #

    # valid
    if config.valid_data_path != '':
        total_scores = []
        with paddle.no_grad():
            for test_data_batch in vaild_annotate_loader:
                feed_input = build_feed_dict(test_data_batch, combine)
                score = model.get_scores(feed_input)
                score = score.cpu().detach().numpy().tolist()
                total_scores += score

        result_dict_ann = evaluate_all_metric(
            qid_list=vaild_annotate_dataset.total_qids,
            label_list=vaild_annotate_dataset.total_labels,
            score_list=total_scores,
            freq_list=vaild_annotate_dataset.total_freqs
        )
        valid_dict['final'] = {
            'dcg@10': result_dict_ann["all_dcg@10"],
            'high': result_dict_ann["high_dcg@10"],
            'mid': result_dict_ann["mid_dcg@10"],
            'low': result_dict_ann["low_dcg@10"],
            'pnr': result_dict_ann["pnr"],
        }
        print(
            f'{idx}th step valid annotate | '
            f'@10 dcg: all {result_dict_ann["all_dcg@10"]:.6f} | '
            f'high {result_dict_ann["high_dcg@10"]:.6f} | '
            f'mid {result_dict_ann["mid_dcg@10"]:.6f} | '
            f'low {result_dict_ann["low_dcg@10"]:.6f} | '
            f'err {result_dict_ann["all_err@10"]:.6f} | '
            f'pnr {result_dict_ann["pnr"]:.6f}'
        )

    # test annotate

    total_scores = []
    with paddle.no_grad():
        for test_data_batch in test_annotate_loader:
            feed_input = build_feed_dict(test_data_batch, combine)
            score = model.get_scores(feed_input)
            score = score.cpu().detach().numpy().tolist()
            total_scores += score

    result_dict_test = evaluate_all_metric(
        qid_list=test_annotate_dataset.total_qids,
        label_list=test_annotate_dataset.total_labels,
        score_list=total_scores,
        freq_list=test_annotate_dataset.total_freqs
    )
    test_dict['final'] = {
        'dcg@10': result_dict_test["all_dcg@10"],
        'high': result_dict_test["high_dcg@10"],
        'mid': result_dict_test["mid_dcg@10"],
        'low': result_dict_test["low_dcg@10"],
        'pnr': result_dict_test["pnr"],
    }
    print(
        f'{idx}th step test annotate | '
        f'@10 dcg: all {result_dict_test["all_dcg@10"]:.6f} | '
        f'high {result_dict_test["high_dcg@10"]:.6f} | '
        f'mid {result_dict_test["mid_dcg@10"]:.6f} | '
        f'low {result_dict_test["low_dcg@10"]:.6f} | '
        f'err {result_dict_test["all_err@10"]:.6f} | '
        f'pnr {result_dict_test["pnr"]:.6f}'
    )
    print('model:', exp_settings['method_name'],
          'total time:', time.time()-start, 'idx:', idx)
    print('save model')
    paddle.save(model.state_dict(), config.output_name + '/{}_{}_{}_total_{:.5f}.model'.format(
        exp_settings['method_name'], change_label, combine, result_dict_test['all_dcg@10']))
