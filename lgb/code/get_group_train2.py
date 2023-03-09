import numpy as np
import random
import lightgbm as lgb
import collections
import pickle
from sklearn.model_selection import cross_val_score
import re


def get_group_new(qid):
    group = []
    query_num = 1
    qid_prev = qid[0]
    doc_num = 1
    for i in range(len(qid)):
        if (qid[i]==qid_prev):
            doc_num += 1
            if (i==0):
                doc_num = 1
            if (i==len(qid)-1):
                group.append(doc_num)
        else:
            query_num += 1
            qid_prev=qid[i]
            group.append(doc_num)
            doc_num = 1

    return group
qid_train = np.load('qid_train_2.npy') 
print(len(qid_train))
group_train = get_group_new(qid_train)
print(len(group_train))

np.save('group_train_2', group_train)

qid_dev = np.load('qid_dev_2.npy') 
print(len(qid_dev))
group_dev = get_group_new(qid_dev)
print(len(group_dev))
np.save('group_dev_2', group_dev)