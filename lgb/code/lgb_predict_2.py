import numpy as np
import random
import lightgbm as lgb
import collections
import pickle
from sklearn.model_selection import cross_val_score
import re
feature_test = np.load('feature/feature_test.npy')
print(feature_test.shape)
print('加载模型用于预测')
bst = lgb.Booster(model_file='lgbmodel_2.txt')

# 预测
y_pred = bst.predict(feature_test)
with open("task2_final_result.csv", "w") as f:
    f.writelines("\n".join(map(str, y_pred)))