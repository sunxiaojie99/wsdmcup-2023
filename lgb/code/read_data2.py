import numpy as np
import random
import lightgbm as lgb
import collections
import pickle
from sklearn.model_selection import cross_val_score
import re
import time


def read_from_file_add(feature_path, data_path, predict_path, featnum=25):
    qid = []
    label = []
    freq = []
    num_of_file = 0
    with open(data_path, 'r') as f:
        for line in f:
            line = re.split('\t', line.strip("\n"))
            qid.append(line[0])
            label.append(line[4])
            if 0 <= int(line[5]) <= 2:  # high freq
                freq.append(0)
            elif 3 <= int(line[5]) <= 6:  # mid freq
                freq.append(1)
            elif 7 <= int(line[5]):  # tail
                freq.append(2)
            # freq.append(line[5])
            num_of_file += 1
    feature = np.zeros((num_of_file, featnum), dtype=float)

    write_num = 0
    with open(feature_path, 'r') as f:
        for line in f:
            line = line.strip("\n").split(" ")
            for numbers in range(len(line)):
                feature[write_num][numbers] = float(line[numbers])
            write_num += 1
    write_num = 0
    with open(predict_path, 'r') as f:
        for line in f:
            write_num += 1
            line = line.strip("\n")
            feature[write_num - 1][featnum - 1] = float(line)
    fix_length = len(qid)
    print("fix_length:", fix_length)
    feature2 = []
    for i in range(fix_length):
        if (freq[i] == 0):
            qid.append(qid[i])
            label.append(label[i])
            feature2.append(feature[i][:])
    feature = np.concatenate((feature, np.array(feature2)))

    return qid, label, feature, freq


def read_from_file(feature_path, data_path, predict_path, featnum=25):
    qid = []
    label = []
    freq = []
    num_of_file = 0
    with open(data_path, 'r') as f:
        for line in f:
            line = re.split('\t', line.strip("\n"))
            qid.append(line[0])
            label.append(line[4])
            if 0 <= int(line[5]) <= 2:  # high freq
                freq.append(0)
            elif 3 <= int(line[5]) <= 6:  # mid freq
                freq.append(1)
            elif 7 <= int(line[5]):  # tail
                freq.append(2)

            num_of_file += 1
    feature = np.zeros((num_of_file, featnum), dtype=float)

    write_num = 0
    with open(feature_path, 'r') as f:
        for line in f:
            line = line.strip("\n").split(" ")
            for numbers in range(len(line)):
                feature[write_num][numbers] = float(line[numbers])
            write_num += 1
    write_num = 0
    with open(predict_path, 'r') as f:
        for line in f:
            write_num += 1
            line = line.strip("\n")
            feature[write_num - 1][featnum - 1] = float(line)
    fix_length = len(qid)

    return qid, label, feature, freq


def create_group(qid):
    group_old = qid[0]
    group = []
    for i in range(len(qid)):
        # 对每一查询qid，如果其值与rand[cnt]相同，则返回其下标x，集合为index
        index = [i for i, x in enumerate(qid) if x == qid[i]]
        group.append(len(index))
    return group


start = time.time()

feature_path = "feature/new_train_features.txt"
data_path = "data/finetine_train.txt"
predict_path = "data/base2-pair-bs352-lr2e-7-neg10/finetine_train.txt"

qid_train, label_train, feature_train, freq_train = read_from_file_add(feature_path, data_path, predict_path,
                                                                       featnum=43)
print(len(qid_train), len(label_train), len(feature_train), len(freq_train))
np.save('qid_train_2', qid_train)
np.save('label_train_2', label_train)
np.save('feature_train_2', feature_train)
np.save('freq_train_2', freq_train)

feature_path_dev = "feature/new_dev_features.txt"
data_path_dev = "data/finetune_dev.txt"
predict_path_dev = "data/base2-pair-bs352-lr2e-7-neg10/finetune_dev.txt"

qid_dev, label_dev, feature_dev, freq_dev = read_from_file(feature_path_dev, data_path_dev, predict_path_dev,
                                                           featnum=43)
print(len(qid_dev), len(label_dev), len(feature_dev))
np.save('qid_dev_2', qid_dev)
np.save('label_dev_2', label_dev)
np.save('feature_dev_2', feature_dev)
np.save('freq_dev_2', freq_dev)

end = time.time()
print(end - start)
