import numpy as np
# -*- encoding: utf-8 -*-
from tqdm import tqdm
from metrics import *


def read_test_data(test_feature_path):
    total_qids = []
    total_labels = []
    total_freqs = []
    for line in tqdm(open(test_feature_path, 'rb')):
        line_list = line.strip(b'\n').split(b'\t')
        qid, query, title, content, label, freq = line_list
        label = int(label)
        qid = int(qid)
        freq = int(freq)
        if 0 <= int(freq) <= 2:  # high freq
            freq = 0
        elif 3 <= int(freq) <= 6:  # mid freq
            freq = 1
        elif 7 <= int(freq):  # tail
            freq = 2
        total_qids.append(qid)
        total_labels.append(label)
        total_freqs.append(freq)
    return total_qids, total_labels, total_freqs


def eval(total_scores, total_qids, total_labels, total_freqs):
    result_dict_test = evaluate_all_metric(
        qid_list=total_qids,
        label_list=total_labels,
        score_list=total_scores,
        freq_list=total_freqs
    )
    print(
        f'@10 dcg: all {result_dict_test["all_dcg@10"]:.6f} | '
        f'high {result_dict_test["high_dcg@10"]:.6f} | '
        f'mid {result_dict_test["mid_dcg@10"]:.6f} | '
        f'low {result_dict_test["low_dcg@10"]:.6f} | '
        f'err {result_dict_test["all_err@10"]:.6f} | '
        f'pnr {result_dict_test["pnr"]:.6f}'
    )


def norm_feature(feat):
    return feat


# 1. 读取原文件
test_feature_path = '/ossfs/workspace/wsdm_cup/data/annotate_data/wsdm_test_2_all.txt'
total_qids, total_labels, total_freqs = read_test_data(test_feature_path)


# 2. 读取模型结果
dir_path = '/ossfs/workspace/wsdm_cup/final/'
model_list = [
    ('12l_dla_combine_neg20_replace_list/best_test2.csv', 2),
    ('12l_dla_combine_neg10_replace_list_freq/best_test2.csv', 1),
    ('12l_dla_combine_neg10_replace_list/best_test2.csv', 3),
    ('3L_dla_neg10_jmtitle_max_bs5_14000/best_3l_dla_9_82_test2.csv', 2),
    ('yll/test_result10.csv', 2),
    ('yll/test_result2.csv', 3),
]
total_scores = None
for test_file, alpha in model_list:
    file_name = test_file
    print(file_name, alpha)
    if total_scores is None:
        total_scores = norm_feature(np.loadtxt(dir_path + file_name)) * alpha
    else:
        total_scores += norm_feature(np.loadtxt(dir_path + file_name)) * alpha


eval(total_scores, total_qids, total_labels, total_freqs)

result_path = 'final_test.csv'
if result_path != '':
    with open(result_path, "w") as f:
        f.writelines("\n".join(map(str, total_scores)))
