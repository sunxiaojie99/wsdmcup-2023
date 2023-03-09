import numpy as np
import random
import lightgbm as lgb
import collections
import pickle
from sklearn.model_selection import cross_val_score
import re
import time


def read_from_file_test(feature_path, predict_path, featnum=25):
    num_of_file = 0   
    with open(predict_path,'r') as f:
        for line in f:
            num_of_file += 1       
    print("num of file:", num_of_file)

    feature=np.zeros((num_of_file, featnum), dtype=float)
    write_num = 0
    with open(feature_path,'r') as f:
        for line in f:
            line=line.strip("\n").split(" ")
            for numbers in range(len(line)):
                feature[write_num][numbers] = line[numbers]
            write_num += 1
    write_num = 0
    with open(predict_path,'r') as f:
        for line in f:
            write_num += 1
            line=line.strip("\n")
            feature[write_num-1][featnum-1] = float(line)
            
    return feature

    
start = time.time()
feature_path = "feature/new_test2_features.txt"
data_path = "data/wsdm_test_2_all.txt"
predict_path = "data/base2-pair-bs352-lr2e-7-neg10/wsdm_test_2_all.txt"

feature_test = read_from_file_test(feature_path, predict_path, featnum=43)
np.save('feature/feature_test', feature_test)
print(feature_test.shape)
print(len(feature_test))

end = time.time()
print(end-start)