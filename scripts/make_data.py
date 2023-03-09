from tqdm import tqdm
import numpy as np

fpath='/ossfs/workspace/wsdm_cup/data/train_data/test_data.txt'
print('load annotated data from ', fpath)

qrels = {}
for line in tqdm(open(fpath, 'rb')):
    line_list = line.strip(b'\n').split(b'\t')
    qid, query, title, content, label, freq = line_list
    qid = int(qid)
    if qid not in qrels:
        qrels[qid] = []
    qrels[qid].append(line)
print('query 数量：', len(qrels))  # 397572 -> query 数量： 5201
np.random.seed(42)  # 保证每次第一次np.random.permutation得到的结果一致
shuffled_indices = np.random.permutation(list(qrels.keys()))  # 生成和原数据等长的无序索引
dev_indices = shuffled_indices[:500]
train_indices = shuffled_indices[500:]
f_out_dev = open(
        '/ossfs/workspace/wsdm_cup/data/new_train/finetune_dev.txt', 'wb')
f_out_train = open(
    '/ossfs/workspace/wsdm_cup/data/new_train/finetine_train.txt', 'wb')

for qid, docs in qrels.items():
    if qid in dev_indices:
        f = f_out_dev
    elif qid in train_indices:
        f = f_out_train
    for line in docs:
        f.write(line)
f_out_dev.close()
f_out_train.close()

"""
query 数量： 5201

1000
finetine_train.txt 319808
finetune_dev.txt: 77764

500
finetine_train.txt 359072
finetune_dev.txt: 38500
"""
