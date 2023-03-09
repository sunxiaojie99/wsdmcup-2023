import gzip
from math import log
import time
import numpy as np

# 只需要考虑query中的word


def cal_df(query, title, content, doc):
    for word in query:
        if word in title:
            try:
                title_df[word] += 1
            except:
                title_df[word] = 1
        if word in content:
            try:
                content_df[word] += 1
            except:
                content_df[word] = 1
        if word in doc:
            try:
                doc_df[word] += 1
            except:
                doc_df[word] = 1

# used for smooth cnt(w,C)


def cal_cnt(words, cnt):
    for word in words:
        try:
            cnt[word] += 1
        except:
            cnt[word] = 1

# BM25中的idf-smoothing


def cal_idf(df):
    return log((total_doc-df+0.5)/(df+0.5), 2)

# 对valid/test提取特征时 可能碰到未出现过的词


def cal_item_feature(query, words, df, qtokens, c_cnt, avg_len):
    '''
    return sum_tf,sum_df,sum_tfidf,length,bm25,ABS,DIR,JM
    (q∩d):tf,bm25,tfidf,ABS,DIR,JM
    only q: idf
    '''
    sum_tf, sum_idf, sum_tfidf, bm25 = 0, 0, 0, 0
    cur_len, u_len, c_len = len(words), 0, sum(list(c_cnt.values()))
    total_w = len(list(c_cnt.keys()))

    # optimal parameters
    lam = 0.1
    miu = 2000
    delta = 0.7
    k1, k2, b = 1.2, 200, 0.75

    # cal cn(word in words) first {word:cnt}
    cnt = {}
    for word in words:
        try:
            cnt[word] += 1
        except:
            cnt[word] = 1
    # cal qcnt for bm25
    qcnt = {}
    for q in query:
        try:
            qcnt[q] += 1
        except:
            qcnt[q] = 1

    # cal u_len
    for word in words:
        try:
            if cnt[word] == c_cnt[word]:
                u_len += cnt[word]
        # valid/test's unique
        except:
            u_len += cnt[word]
    if u_len == 0:
        u_len = 1
    # cal alpha for three methods
    ABS_alpha = delta*u_len/cur_len  # can be zero=>
    DIR_alpha = miu/(miu+cur_len)
    ABS, DIR, JM = len(query)*log(ABS_alpha, 2), len(query) * \
        log(DIR_alpha, 2), len(query)*log(lam, 2)
    for q in query:
        idf = cal_idf(df[q] if q in qtokens else 0)
        sum_idf += idf
        try:
            p = (c_cnt[q]+1)/(c_len+total_w)  # p(w|C)
        except:
            p = 1/(c_len+total_w)
        unseenp_JM, unseenp_DIR, unseenp_ABS = lam*p, DIR_alpha*p, ABS_alpha*p
        # seen in d
        if q in words:
            tf = cnt[q]/cur_len
            qtf = qcnt[q]/len(query)
            K = k1*(1-b+b*cur_len/avg_len)
            sum_tf += tf
            sum_tfidf += tf*idf
            bm25 += idf*((k1+1)*tf/(K+tf))*((k2+1)*qtf/(k2+qtf))
            seenp_JM = (1-lam)*tf+unseenp_JM
            seenp_DIR = cnt[q]/(cur_len+miu)+unseenp_DIR
            seenp_ABS = max(0, cnt[q]-delta)/cur_len+unseenp_ABS
            JM += log(seenp_JM/unseenp_JM, 2)
            DIR += log(seenp_DIR/unseenp_DIR, 2)
            ABS += log(seenp_ABS/unseenp_ABS, 2)

    return [sum_tf, sum_idf, sum_tfidf, cur_len, bm25, JM, DIR, ABS]


def cal_total_feature(query, title, content, doc):

    features = []

    features += cal_item_feature(query, title,
                                 title_df, qtitle_tokens, title_cnt, titles_len)
    features += cal_item_feature(query, content, content_df,
                                 qcontent_tokens, content_cnt, contents_len)
    features += cal_item_feature(query, doc, doc_df,
                                 qdoc_tokens, doc_cnt, docs_len)

    return features

# normalize features with Gaussion


def normalize_features(features):
    features = np.array(features)
    m = features.mean(axis=0)
    s = features.std(axis=0)
    s[s == 0] = 1
    print(m)
    print(s)
    return (features-m)/s


# {q-word:df}
doc_df, title_df, content_df = {}, {}, {}
# {word: cnt}
doc_cnt, title_cnt, content_cnt = {}, {}, {}
total_doc, docs_len, titles_len, contents_len = 0, 0, 0, 0
# calculate word_df
with gzip.open('/ossfs/workspace/wsdm_cup/data/train_data/part-00001.gz', 'rb') as f:
    for line in f.readlines():
        line_list = line.strip(b'\n').split(b'\t')
        if len(line_list) == 3:
            query = line_list[1]
            query = query.split(b'\x01')
        else:
            total_doc += 1
            title, content = line_list[2], line_list[3]
            title = title.split(b'\x01')
            content = content.split(b'\x01')
            doc = title+content
            docs_len += len(doc)
            titles_len += len(title)
            contents_len += len(content)
            cal_df(query, title, content, doc)
            cal_cnt(doc, doc_cnt)
            cal_cnt(title, title_cnt)
            cal_cnt(content, content_cnt)
# avg_len
docs_len /= total_doc
titles_len /= total_doc
contents_len /= total_doc
qdoc_tokens, qtitle_tokens, qcontent_tokens = list(
    doc_df.keys()), list(title_df.keys()), list(content_df.keys())


def train():
    idx = 0
    train_features = []
    # calculate features for train
    with gzip.open('/ossfs/workspace/wsdm_cup/data/train_data/part-00001.gz', 'rb') as f:
        for line in f.readlines():
            idx += 1
            print('train', idx)
            line_list = line.strip(b'\n').split(b'\t')
            if len(line_list) == 3:
                query = line_list[1]
                query = query.split(b'\x01')
            else:
                title, content = line_list[2], line_list[3]
                title = title.split(b'\x01')
                content = content.split(b'\x01')
                doc = title+content
                train_features.append(
                    cal_total_feature(query, title, content, doc))
    print('save train_features')
    np.savetxt('/ossfs/workspace/wsdm_cup/data/new_train/features/part-00001.txt',
               np.array(train_features), fmt='%f', delimiter=' ')
    train_features = normalize_features(train_features)
    np.savetxt('/ossfs/workspace/wsdm_cup/data/new_train/features/norm_part-00001.txt',
               np.array(train_features), fmt='%f', delimiter=' ')


def test():
    idx = 0
    last_features = []
    # calculate features for valid=>p(w|C) df用的是训练集的数据
    with open('/ossfs/workspace/wsdm_cup/data/annotate_data/wsdm_test_2_all.txt', 'rb') as f:
        for line in f.readlines():
            idx += 1
            print('real test', idx)
            line_list = line.strip(b'\n').split(b'\t')
            _, query, title, content, _, _ = line_list
            query = query.split(b'\x01')
            title = title.split(b'\x01')
            content = content.split(b'\x01')
            doc = title+content
            last_features.append(cal_total_feature(query, title, content, doc))
    last_features = normalize_features(last_features)
    print('save real test features')
    np.savetxt('/ossfs/workspace/wsdm_cup/data/new_train/features/wsdm_test_2_all.txt',
               np.array(last_features), fmt='%f', delimiter=' ')


def finetine_train():
    idx = 0
    valid_features = []
    with open('/ossfs/workspace/wsdm_cup/data/new_train/finetine_train.txt', 'rb') as f:
        for line in f.readlines():
            idx += 1
            print('valid', idx)
            line_list = line.strip(b'\n').split(b'\t')
            _, query, title, content, _, _ = line_list
            query = query.split(b'\x01')
            title = title.split(b'\x01')
            content = content.split(b'\x01')
            doc = title+content
            valid_features.append(
                cal_total_feature(query, title, content, doc))
    valid_features = normalize_features(valid_features)
    print('save valid_features')
    np.savetxt('/ossfs/workspace/wsdm_cup/data/new_train/features/finetine_train.txt',
               np.array(valid_features), fmt='%f', delimiter=' ')


def finetune_dev():
    idx = 0
    test_features = []
    with open('/ossfs/workspace/wsdm_cup/data/new_train/finetune_dev.txt', 'rb') as f:
        for line in f.readlines():
            idx += 1
            print('test', idx)
            line_list = line.strip(b'\n').split(b'\t')
            _, query, title, content, _, _ = line_list
            query = query.split(b'\x01')
            title = title.split(b'\x01')
            content = content.split(b'\x01')
            doc = title+content
            test_features.append(cal_total_feature(query, title, content, doc))
    test_features = normalize_features(test_features)
    print('save test_features')
    np.savetxt('/ossfs/workspace/wsdm_cup/data/new_train/features/finetune_dev.txt',
               np.array(test_features), fmt='%f', delimiter=' ')

train()
test()
finetine_train()
finetune_dev()
