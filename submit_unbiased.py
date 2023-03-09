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
import datetime
import os
import matplotlib.pyplot as plt

random.seed(config.seed+1)
random.seed(config.seed)
np.random.seed(config.seed)

paddle.set_device("gpu:0")
paddle.seed(config.seed)

print(config)
exp_settings = config.exp_settings
combine = config.combine


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
    projection=config.projection,
    rank_feature_size=config.exp_settings['rank_feature_size']
)
# load pretrained model
if config.init_parameters != "":
    print('load warm up model ', config.init_parameters)
    ptm = paddle.load(config.init_parameters)
    if 'model' in ptm:
        ptm = ptm['model']
    for k, v in token_encoder.state_dict().items():
        if not k in ptm:
            pass
            print("warning: not loading " + k)
        else:
            print("loading " + k)
            v.set_value(ptm[k])


# load pretrained model=>baseline_model中load

method_str = exp_settings['method_name']
if method_str not in ['IPWrank', 'DLA', 'RegressionEM', 'PairDebias', 'NaiveAlgorithm']:
    print("please choose a method in 'IPWrank', 'DLA','RegressionEM','PairDebias','NaiveAlgorithm'")
    sys.exit()
model = find_class('baseline_model.learning_algorithm.' +
                   method_str)(exp_settings=exp_settings, encoder_model=token_encoder)

test_annotate_dataset = TestDataset(config.test_data_path, max_seq_len=config.max_seq_len,
                                    data_type='task1_annotate', load_feature=combine, feature_path=config.test_feature_path)
test_annotate_loader = DataLoader(
    test_annotate_dataset, batch_size=config.eval_batch_size)

idx = -1
# test annotate
t0 = time.time()
total_scores = []
with paddle.no_grad():
    for test_data_batch in test_annotate_loader:
        feed_input = build_feed_dict(test_data_batch, combine)
        score = model.get_scores(feed_input)
        score = score.cpu().detach().numpy().tolist()
        total_scores += score

if config.result_path != '':
    with open(config.result_path, "w") as f:
        f.writelines("\n".join(map(str, total_scores)))

result_dict_test = evaluate_all_metric(
    qid_list=test_annotate_dataset.total_qids,
    label_list=test_annotate_dataset.total_labels,
    score_list=total_scores,
    freq_list=test_annotate_dataset.total_freqs
)
print(
    f'{idx}th step test annotate | '
    f'@10 dcg: all {result_dict_test["all_dcg@10"]:.6f} | '
    f'high {result_dict_test["high_dcg@10"]:.6f} | '
    f'mid {result_dict_test["mid_dcg@10"]:.6f} | '
    f'low {result_dict_test["low_dcg@10"]:.6f} | '
    f'err {result_dict_test["all_err@10"]:.6f} | '
    f'pnr {result_dict_test["pnr"]:.6f}'
)
t1 = time.time()
training_time = t1 - t0


def format_time(time):
    elapsed_rounded = int(round((time)))
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


training_time = format_time(training_time)
print('eval_time', training_time)
