import numpy as np
import random
import lightgbm as lgb
import collections
import pickle
from sklearn.model_selection import cross_val_score
import re
import time
import datetime

def format_time(time):
    elapsed_rounded = int(round((time)))
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
t0 = time.time()

qid_train = np.load('qid_train_2.npy')
label_train = np.load('label_train_2.npy')
feature_train = np.load('feature_train_2.npy')
group_train = np.load('group_train_2.npy')

qid_dev = np.load('qid_dev_2.npy')
label_dev = np.load('label_dev_2.npy')
feature_dev = np.load('feature_dev_2.npy')
group_dev = np.load('group_dev_2.npy')

# 训练过程中参数设置
param = {
    'task': 'train', 
    'boosting_type': 'gbdt', 
    'objective': 'lambdarank',  
    'metric': 'ndcg',  
    'max_position': 10,  
    'metric_freq': 100,  
    'train_metric': False, 
    'ndcg_at': [10], 
    'max_bin': 255,  
    'num_iterations': 4000, 
    'learning_rate': 0.005, 
    'num_leaves': 31, 
    'max_depth': 6,
    'feature_fraction': 0.8,  
    'tree_learner': 'serial',  
    'min_data_in_leaf': 30, 
    'verbose': 2 
}

res = {}

def get_groups(qids):
    prev_qid = None
    prev_limit = 0
    total = 0
    for i, qid in enumerate(qids):
        total += 1
        if qid != prev_qid:
            if i != prev_limit:
                yield (prev_qid, prev_limit, i)
            prev_qid = qid
            prev_limit = i

        if prev_limit != total:
            yield (prev_qid, prev_limit, total)


def dcg_k(scores, k):
    return np.sum([
        (np.power(2, float(scores[i])) - 1) / np.log2(i + 2)
        for i in range(len(scores[:k]))
    ])


def ideal_dcg_k(scores, k):
    # 相关度降序排序
    scores = [score for score in sorted(scores)[::-1]]
    return dcg_k(scores, k)


def validate(qids, targets, preds, k):
    query_groups = get_groups(qids)  
    all_ndcg = []
    all_dcg = []
    every_qid_ndcg = collections.OrderedDict()

    for qid, a, b in query_groups:
        predicted_sorted_indexes = np.argsort(preds[a:b])[::-1]  # 从大到小的索引
        t_results = targets[a:b]  # 目标数据的相关度
        t_results = np.array(t_results)[predicted_sorted_indexes]  # 是predicted_sorted_indexes排好序的在test_data中的相关度
        dcg_val = dcg_k(t_results, k)
        idcg_val = ideal_dcg_k(t_results, k)
        ndcg_val = (dcg_val / idcg_val)
        all_dcg.append(dcg_val)
        all_ndcg.append(ndcg_val)
        every_qid_ndcg.setdefault(qid, ndcg_val)

    average_ndcg = np.nanmean(all_ndcg)
    average_dcg = np.nanmean(all_dcg)
    return average_ndcg, every_qid_ndcg, average_dcg


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variable(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


train_feature = lgb.Dataset(feature_train, label=label_train, group=group_train)
bst = lgb.train(param, train_feature, num_boost_round=10, evals_result=res, verbose_eval=10)
bst.save_model('lgbmodel_2.txt')

dev_feature = lgb.Dataset(feature_dev, label=label_dev, group=group_dev)
ypred = bst.predict(feature_dev)
average_ndcg10, _, avg_dcg10 = validate(qid_dev, label_dev, ypred, 10)

with open("lgb_ndcg_2.txt", "a") as f:
    f.write("\n")
    f.write(str(param))
    f.write('\n')

    f.write("average 10@dcg: ")
    f.write(str(avg_dcg10))
    f.write("\n")

print("average 10@dcg: ", avg_dcg10)
t1 = time.time()
training_time = t1 - t0
training_time = format_time(training_time)
print('training_time', training_time)