# -*- encoding: utf-8 -*-
'''
@Time    :   2022/06/12 14:49:28
@Author  :   Chu Xiaokai 
@Contact :   xiaokaichu@gmail.com
'''
import numpy as np
import warnings
import sys
from Transformer4Ranking.model import *
from paddle.io import DataLoader
from dataloader import *
from args import config
from tqdm import tqdm
import time
import datetime
from metrics import evaluate_all_metric

random.seed(config.seed+1)
random.seed(config.seed)
np.random.seed(config.seed)
paddle.set_device("gpu:0")
paddle.seed(config.seed)
print(config)
exp_settings = config.exp_settings

model = TransformerModel(
    ntoken=config.ntokens,
    hidden=config.emb_dim,
    nhead=config.nhead,
    nlayers=config.nlayers,
    dropout=config.dropout,
    mode=config.finetune_type
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

test_annotate_dataset = TestDataset(
    config.test_annotate_path, max_seq_len=config.max_seq_len, data_type='annotate', config=config)
test_annotate_loader = DataLoader(
    test_annotate_dataset, batch_size=config.eval_batch_size)
# evaluate
total_scores = []

t0 = time.time()
for test_data_batch in test_annotate_loader:
    model.eval()
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
print(
    f'valid annotate | '
    f'dcg@10: all {result_dict_ann["all_dcg@10"]:.6f} | '
    f'high {result_dict_ann["high_dcg@10"]:.6f} | '
    f'mid {result_dict_ann["mid_dcg@10"]:.6f} | '
    f'low {result_dict_ann["low_dcg@10"]:.6f} | '
    f'pnr {result_dict_ann["pnr"]:.6f}'
)
t1 = time.time()
training_time = t1 - t0


def format_time(time):
    elapsed_rounded = int(round((time)))
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


training_time = format_time(training_time)
print('eval_time', training_time)

if config.result_path is not None:
    with open(config.result_path, "w") as f:
        f.writelines("\n".join(map(str, total_scores)))
