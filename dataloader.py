# -*- encoding: utf-8 -*-
'''
@Time    :   2022/06/10 15:51:44
@Author  :   Chu Xiaokai
@Contact :   xiaokaichu@gmail.com
'''
import math
import paddle
import paddle.nn.functional as F
import os
import random
from paddle.io import Dataset, IterableDataset
import gzip
from functools import reduce
from args import config
import numpy as np
from tqdm import tqdm
from math import log

# feature type=>idx


def type_idx(type):
    mapping = {}
    features = []
    for item in ['title', 'content', 'title_content']:
        for feature in ['tf', 'idf', 'tfidf', 'len', 'bm25', 'JM', 'DIR', 'ABS']:
            features.append('{}_{}'.format(feature, item))
    for idx, feature in enumerate(features):
        mapping[feature] = idx
    mapping['query_len'] = 24
    return mapping[type]


def vote(feature):

    def norm_feature(feat):
        return feat
        # feat_min = feat.min()
        # feat_max = feat.max()
        # norm_feat = (feat - feat_min) / (feat_max - feat_min)
        # return norm_feat

    feat_list = ['JM_title', 'ABS_title',
                 'tfidf_title', 'JM_title_content',
                 'ABS_title_content', 'ABS_content']
    alpha_list = [1, 1, 1, 1, 1, 1]

    res = None
    for idx, feat in enumerate(feat_list):
        if res is None:
            res = norm_feature(
                feature[:, type_idx(feat)]) * alpha_list[idx]
        else:
            res += norm_feature(
                feature[:, type_idx(feat)]) * alpha_list[idx]
    return res


def vote_for_neg(feature):
    feat_list = ['JM_title', 'ABS_title',
                 'tfidf_title', 'JM_title_content',
                 'ABS_title_content', 'ABS_content']
    alpha_list = [1, 1, 1, 1, 1, 1]

    res = None
    for idx, feat in enumerate(feat_list):
        if res is None:
            res = feature[type_idx(feat)] * alpha_list[idx]
        else:
            res = feature[type_idx(feat)] * alpha_list[idx]
    return res

# --------------- auxiliary functions for statistics and features----------------  #
# used for smooth cnt(w,C)


def cal_cnt(words, cnt):
    for word in words:
        try:
            cnt[word] += 1
        except:
            cnt[word] = 1

# BM25中的idf-smoothing


def cal_idf(df, total_doc):
    return log((total_doc-df+0.5)/(df+0.5), 2)


def cal_item_feature(query, words, df, qtokens, c_cnt, avg_len, total_doc):
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
        idf = cal_idf(df[q] if q in qtokens else 0, total_doc)
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

# --------------- data process for  masked language modeling (MLM) ----------------  #


def prob_mask_like(t, prob):
    return paddle.to_tensor(paddle.zeros_like(t), dtype=paddle.float32).uniform_(0, 1) < prob


def mask_with_tokens(t, token_ids):
    """
    t: size [bs, seq_len]
    token_ids: [0, 1, 2] 不需要考虑的token id list

    mask: [bs, seq_len], 在mlm中不能被mask的位置是true，可以被mask的地方是false
    """
    init_no_mask = paddle.full_like(
        t, False, dtype=paddle.bool)  # 全false, [bs, seq_len]
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    # 用init_no_mask全fasle作初始化，逐个用不需要被考虑的id对全batch的token id进行判断，如果相等就是true，
    # 所有特殊的token id 得到的布尔矩阵进行求或
    return mask


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)  # [bs, seq_len] 全 -1e9
    # 如果一个位置可以被mlm mask，就填充value -1e9，否则填充x（随机）
    return paddle.where(mask, y, x)


def get_mask_subset_with_prob(mask, prob):
    """
    mask: [bs, seq_len] 可以被mlm mask的地方是true
    prob: 0.1
    """
    batch, seq_len = mask.shape
    # 每行最多mask多少个token(无padding), 0.1*128=13, 向上取整
    max_masked = math.ceil(prob * seq_len)
    num_tokens = paddle.sum(mask, axis=-1, keepdim=True)  # 每个样本可以被mask的数量
    # to check
    # paddle.cumsum(paddle.to_tensor(mask, dtype="int32"), axis=-1) 按行，每个元素等于该元素之前同行有多少个有效的token
    """
    mask: [False, True , True , True , False, False]
    cumsum: [0, 1, 2, 3, 3, 3]
    num_tokens * prob: 每行最多被mask的token个数
    mask_excess： [bs, seq_len], true代表这一行此位置及之前已经够了这一行被mask掉的个数

    """
    # true代表这一行此位置及之前已经够了这一行被mask掉的个数
    mask_excess = (paddle.cumsum(paddle.to_tensor(
        mask, dtype="int32"), axis=-1) > (num_tokens * prob).ceil())

    # [bs, max_masked_num] 截取最多被mask的个数
    mask_excess = mask_excess[:, :max_masked]

    # 可以被mlm mask的位置是-1e9，不可以被mask的位置是0-1之间随机数
    rand = masked_fill(paddle.rand((batch, seq_len)),
                       mask, -1e9)  # [bs, seq_len]

    # 对rand数组取topk的位置，也就是topk个随机数？
    _, sampled_indices = rand.topk(max_masked, axis=-1)
    sampled_indices = masked_fill(
        sampled_indices + 1, mask_excess, 0)  # [bs, 13]

    new_mask = paddle.zeros((batch, seq_len + 1))
    rows = paddle.reshape(paddle.to_tensor(np.array([[i] * max_masked for i in range(batch)]),
                                           dtype=paddle.int64), (-1,))
    cols = paddle.reshape(sampled_indices, (-1,))
    new_mask[rows, cols] = 1
    return paddle.to_tensor(new_mask[:, 1:], dtype=paddle.bool)


def mask_data(seq, mask_ignore_token_ids=[config._CLS_, config._SEP_, config._PAD_],
              mask_token_id=config._MASK_,
              mask_prob=0.1,
              pad_token_id=config._PAD_,
              replace_prob=1.0
              ):
    """
    seq: [bs, seq_len]
    mask_ignore_token_ids: 在mlm mask的时候忽略的token id, eg, [0, 1, 2]
    mask_token_id: 3
    pad_token_id: 2
    """
    no_mask = mask_with_tokens(
        seq, mask_ignore_token_ids)  # [150, 128] mlm不需要考虑的地方，为True
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)
    masked_seq = seq.clone()
    # [bs, seq_len], mlm的label, mask为false的位置填充原始的token id，true的填2
    labels = masked_fill(seq, mask, pad_token_id)
    # seq.masked_fill(~mask, pad_token_id)  # use pad to fill labels
    replace_prob = prob_mask_like(seq, replace_prob)
    mask = mask * replace_prob
    masked_seq = masked_fill(masked_seq, mask, mask_token_id)
    return masked_seq, labels


# ----------------------  DataLoader ----------------------- #
def process_data(query, title, content, max_seq_len):
    """ process [query, title, content] into a tensor 
        [CLS] + query + [SEP] + title + [SEP] + content + [SEP] + [PAD]

        generate features(24) for q-d(title content title+content)
    """
    data = [config._CLS_]
    segment = [0]

    data = data + [int(item) + 10 for item in query.split(b'\x01')]  # query
    data = data + [config._SEP_]
    # 区分query和doc，为sep多加一个
    segment = segment + [0] * (len(query.split(b'\x01')) + 1)

    data = data + [int(item) + 10 for item in title.split(b'\x01')]  # title
    data = data + [config._SEP_]  # sep defined as 1
    segment = segment + [1] * (len(title.split(b'\x01')) + 1)

    data = data + \
        [int(item) + 10 for item in content.split(b'\x01')]  # content
    data = data + [config._SEP_]
    segment = segment + [1] * (len(content.split(b'\x01')) + 1)

    # padding
    padding_mask = [False] * len(data)  # 默认非填充
    if len(data) < max_seq_len:
        padding_mask += [True] * (max_seq_len - len(data))
        data += [config._PAD_] * (max_seq_len - len(data))
    else:
        padding_mask = padding_mask[:max_seq_len]
        data = data[:max_seq_len]

    # segment id
    if len(segment) < max_seq_len:
        segment += [1] * (max_seq_len-len(segment))
    else:
        segment = segment[:max_seq_len]
    padding_mask = paddle.to_tensor(padding_mask, dtype='int32')
    data = paddle.to_tensor(data, dtype="int32")
    segment = paddle.to_tensor(segment, dtype="int32")
    return data, segment, padding_mask


class TrainDataset(IterableDataset):
    def __init__(self, directory_path, buffer_size=100000, max_seq_len=128):
        super().__init__()
        self.directory_path = directory_path
        self.buffer_size = buffer_size
        self.files = os.listdir(self.directory_path)
        random.shuffle(self.files)
        self.cur_query = "#"
        self.max_seq_len = max_seq_len

    def __iter__(self):
        buffer = []
        for file in self.files:
            print('load file', file)
            if file[-3:] != '.gz' or file == 'part-00000.gz':  # part-00000.gz is for evaluation
                continue
            with gzip.open(os.path.join(self.directory_path, file), 'rb') as f:
                for line in f.readlines():
                    line_list = line.strip(b'\n').split(b'\t')
                    if len(line_list) == 3:  # new query
                        self.cur_query = line_list[1]
                    elif len(line_list) > 6:  # urls
                        position, title, content, click_label = line_list[
                            0], line_list[2], line_list[3], line_list[5]
                        try:
                            src_input, segment, src_padding_mask = process_data(
                                self.cur_query, title, content, self.max_seq_len)
                            buffer.append(
                                [src_input, segment, src_padding_mask, float(click_label)])
                        except:
                            pass
                    if len(buffer) >= self.buffer_size:
                        random.shuffle(buffer)

                        for record in buffer:
                            yield record
                        buffer = []


class TestDataset(Dataset):

    def __init__(self, fpath, max_seq_len, data_type, config=None, buffer_size=300000, load_feature=0, feature_path=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        self.load_feature = load_feature
        self.data_type = data_type
        if config:
            self.negative_num = config.negative_num
            self.pos_num = config.pos_num
            self.strategy = config.strategy

        if data_type == 'annotate':
            self.buffer, self.total_qids, self.total_labels, self.total_freqs = self.load_annotate_data(
                fpath, feature_path)
        elif data_type == 'finetune':
            self.buffer, self.total_qids, self.total_labels, self.total_freqs = self.load_annotate_data(
                fpath, shuffle=True, binary_label=True)
        elif data_type == 'finetune_pair':
            self.buffer, self.total_qids, self.total_labels, self.total_freqs, \
                self.total_features = self.load_finetune_data_pairwise(
                    fpath, feature_path, shuffle=True, binary_label=True)
        elif data_type == 'finetune_list':
            self.buffer, self.total_qids, self.total_labels, self.total_freqs = self.load_finetune_list_data(
                fpath, shuffle=True, binary_label=True)
        elif data_type == 'click':
            self.buffer, self.total_qids, self.total_labels = self.load_click_data(
                fpath)
        elif data_type == 'task1_annotate':
            self.buffer, self.total_qids, self.total_labels, self.total_freqs = self.load_data_feature(
                fpath, feature_path)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        if self.data_type == 'finetune_pair' and self.load_feature:
            return self.buffer[index], self.total_features[index]
        return self.buffer[index]

    def load_finetune_list_data(self, fpath, shuffle=False, binary_label=False):
        print('load annotated data from ', fpath)
        idx = 0
        qrels = {}
        for line in tqdm(open(fpath, 'rb')):
            line_list = line.strip(b'\n').split(b'\t')
            if len(line_list) != 6:
                print(idx, line_list)
            qid, query, title, content, label, freq = line_list
            label = int(label)
            qid = int(qid)
            freq = int(freq)
            src_input, src_segment, src_padding_mask = process_data(
                query, title, content, self.max_seq_len)
            if qid not in qrels:
                qrels[qid] = {
                    '4_list': [],
                    '3_list': [],
                    '2_list': [],
                    '1_list': [],
                    '0_list': []
                }
                if len(query) <= 3:
                    print('short query:', line_list)
                if len(title) <= 3:
                    print('short title:', line_list)

            qrels[qid]['{}_list'.format(label)].append(
                [src_input, src_segment, src_padding_mask, label, freq])
            idx += 1
        qids = list(qrels.keys())
        if shuffle:
            random.shuffle(qids)  # 按qid shuffle
        examples = []
        num_samples = 0
        total_qids = []
        total_labels = []
        total_freqs = []
        for qid in tqdm(qids):

            pos_ids_detail = qrels[qid]['4_list'] + \
                qrels[qid]['3_list'] + qrels[qid]['2_list']
            neg_ids_detail = qrels[qid]['1_list'] + qrels[qid]['0_list']

            # all_ids_detail = pos_ids_detail + neg_ids_detail
            # if len(all_ids_detail) == 0:
            #     continue
            # all_ids = [i for i in range(len(all_ids_detail))]
            # need_ids = all_ids[:60]
            # if len(need_ids) < 60:
            #     still_need_num = 60 - len(need_ids)
            #     replacement = False if len(all_ids) > still_need_num else True
            #     still_ids = np.random.choice(
            #         all_ids, still_need_num, replace=replacement)
            #     need_ids.extend(still_ids)
            # assert len(need_ids) == 60
            # examples.extend([all_ids_detail[i][:4] for i in need_ids])
            # total_qids.extend([qid for _ in need_ids])
            # total_labels.extend([all_ids_detail[i][3] for i in need_ids])
            # total_freqs.extend([all_ids_detail[i][4] for i in need_ids])

            # num_samples += 60

            if len(pos_ids_detail) == 0 or len(neg_ids_detail) == 0:  # 没有正例就continue
                continue
            pos_ids = [i for i in range(len(pos_ids_detail))]
            neg_ids = [i for i in range(len(neg_ids_detail))]
            pos_replacement = False if len(
                pos_ids_detail) > self.pos_num else True  # 如果样本数不足，可以重复采样
            neg_replacement = False if len(
                neg_ids_detail) > self.negative_num else True

            pos_ids = np.random.choice(
                pos_ids, self.pos_num, replace=pos_replacement)
            neg_ids = np.random.choice(
                neg_ids, self.negative_num, replace=neg_replacement)

            examples.extend([pos_ids_detail[i][:4] for i in pos_ids])
            examples.extend([neg_ids_detail[i][:4] for i in neg_ids])

            total_qids.extend([qid for _ in pos_ids])
            total_qids.extend([qid for _ in neg_ids])

            total_labels.extend([pos_ids_detail[i][3] for i in pos_ids])
            total_labels.extend([neg_ids_detail[i][3] for i in neg_ids])
            total_freqs.extend([pos_ids_detail[i][4] for i in pos_ids])
            total_freqs.extend([neg_ids_detail[i][4] for i in neg_ids])

            num_samples += self.negative_num + self.pos_num  # 样本数添加对应的数量

        print('Num samples: {}'.format(num_samples))

        return examples, total_qids, total_labels, total_freqs

    def load_finetune_data_pairwise(self, fpath, feature_path=None, shuffle=False, binary_label=False):
        """先针对binary，>=2的看作pos,其余为neg
        todo: pos=4, neg=0,1,2,3
        pos=3, neg=0,1,2 细化分级
        """
        print('load annotated data from ', fpath)
        if self.load_feature:
            print('load features', feature_path)
            annotate_features = np.loadtxt(feature_path)
        idx = 0
        qrels = {}
        for line in tqdm(open(fpath, 'rb')):
            line_list = line.strip(b'\n').split(b'\t')
            if len(line_list) != 6:
                print(idx, line_list)
            qid, query, title, content, label, freq = line_list
            label = int(label)
            qid = int(qid)
            freq = int(freq)
            src_input, src_segment, src_padding_mask = process_data(
                query, title, content, self.max_seq_len)
            if qid not in qrels:
                qrels[qid] = {
                    '4_list': [],
                    '3_list': [],
                    '2_list': [],
                    '1_list': [],
                    '0_list': []
                }
                if len(query) <= 3:
                    print('short query:', line_list)
                if len(title) <= 3:
                    print('short title:', line_list)
            if binary_label:
                if label >= 2:
                    new_label = 1
                else:
                    new_label = 0
            else:
                new_label = label

            if self.load_feature:
                qrels[qid]['{}_list'.format(label)].append(
                    [src_input, src_segment, src_padding_mask,
                        new_label, freq,
                        paddle.to_tensor(
                            annotate_features[idx], dtype=paddle.float32),
                     ])
            else:
                qrels[qid]['{}_list'.format(label)].append(
                    [src_input, src_segment, src_padding_mask, new_label, freq])
            idx += 1
        qids = list(qrels.keys())
        if shuffle:
            random.shuffle(qids)  # 按qid shuffle
        examples = []
        num_samples = 0
        total_qids = []
        total_labels = []
        total_freqs = []
        total_features = []
        for qid in tqdm(qids):
            pos_ids_detail = qrels[qid]['4_list'] + \
                qrels[qid]['3_list'] + qrels[qid]['2_list']
            neg_ids_detail = qrels[qid]['1_list'] + qrels[qid]['0_list']
            if len(pos_ids_detail) == 0 or len(neg_ids_detail) == 0:  # 没有正例就continue
                continue
            if self.strategy == 'q':
                # 对于一个query，每个epoch随机选择1个正例doc_id
                pos_ids = [i for i in range(len(pos_ids_detail))]
                pos_id = np.random.choice(pos_ids)
                pos_detail = pos_ids_detail[pos_id]

                examples.append(pos_detail[:4])
                total_qids.append(qid)
                total_labels.append(pos_detail[3])
                total_freqs.append(pos_detail[4])
                if self.load_feature:
                    total_features.append(pos_detail[5])

                replacement = False if len(
                    neg_ids_detail) > self.negative_num else True  # 如果负样本数不足，可以重复采样
                neg_ids = [i for i in range(len(neg_ids_detail))]
                neg_ids = np.random.choice(
                    neg_ids, self.negative_num, replace=replacement)  # 选出匹配的neg例

                examples.extend([neg_ids_detail[i][:4] for i in neg_ids])
                total_qids.extend([qid for _ in neg_ids])
                total_labels.extend([neg_ids_detail[i][3] for i in neg_ids])
                total_freqs.extend([neg_ids_detail[i][4] for i in neg_ids])
                if self.load_feature:
                    total_features.extend([neg_ids_detail[i][5]
                                           for i in neg_ids])

                num_samples += self.negative_num + 1  # 样本数添加对应的数量
            elif self.strategy == 'd':
                for pos_detail in pos_ids_detail:
                    total_qids.append(qid)
                    total_labels.append(pos_detail[3])
                    total_freqs.append(pos_detail[4])
                    examples.append(pos_detail[:4])
                    if self.load_feature:
                        total_features.append(pos_detail[5])

                    replacement = False if len(
                        neg_ids_detail) > self.negative_num else True  # 如果负样本数不足，可以重复采样
                    neg_ids = [i for i in range(len(neg_ids_detail))]
                    neg_ids = np.random.choice(
                        neg_ids, self.negative_num, replace=replacement)

                    examples.extend([neg_ids_detail[i][:4] for i in neg_ids])
                    total_qids.extend([qid for _ in neg_ids])
                    total_labels.extend([neg_ids_detail[i][3]
                                         for i in neg_ids])
                    total_freqs.extend([neg_ids_detail[i][4] for i in neg_ids])
                    if self.load_feature:
                        total_features.extend(
                            [neg_ids_detail[i][5] for i in neg_ids])

                    num_samples += self.negative_num + 1
            else:
                raise ValueError('invalid strategy to sample data!')
        print('Num samples: {}'.format(num_samples))

        return examples, total_qids, total_labels, total_freqs, total_features

    def load_annotate_data(self, fpath, feature_path=None, shuffle=False, binary_label=False):
        if self.load_feature:
            print('load features', feature_path)
            annotate_features = np.loadtxt(feature_path)
        print('load annotated data from ', fpath)
        total_qids = []
        buffer = []
        total_labels = []
        total_freqs = []
        idx = 0
        for line in tqdm(open(fpath, 'rb')):
            line_list = line.strip(b'\n').split(b'\t')
            if len(line_list) != 6:
                print(idx, line_list)
            qid, query, title, content, label, freq = line_list
            if binary_label:
                if int(label) >= 2:
                    label = "1"
                else:
                    label = "0"
            if 0 <= int(freq) <= 2:  # high freq
                freq = 0
            elif 3 <= int(freq) <= 6:  # mid freq
                freq = 1
            elif 7 <= int(freq):  # tail
                freq = 2
            total_qids.append(int(qid))
            total_labels.append(int(label))
            total_freqs.append(freq)
            src_input, src_segment, src_padding_mask = process_data(
                query, title, content, self.max_seq_len)
            if self.load_feature:
                buffer.append([src_input, src_segment, src_padding_mask, label,
                               paddle.to_tensor(annotate_features[idx], dtype=paddle.float32)])
            else:
                buffer.append(
                    [src_input, src_segment, src_padding_mask, label])
            idx += 1
        if shuffle:
            np.random.shuffle(buffer)
        return buffer, total_qids, total_labels, total_freqs

    def load_click_data(self, fpath):
        print('load logged click data from ', fpath)
        with gzip.open(fpath, 'rb') as f:
            buffer = []
            total_qids = []
            total_labels = []
            cur_qids = 0
            for line in f.readlines():
                line_list = line.strip(b'\n').split(b'\t')
                if len(line_list) == 3:  # new query
                    self.cur_query = line_list[1]
                    cur_qids += 1
                elif len(line_list) > 6:  # urls
                    position, title, content, click_label = line_list[
                        0], line_list[2], line_list[3], line_list[5]
                    try:
                        src_input, src_segment, src_padding_mask = process_data(
                            self.cur_query, title, content, self.max_seq_len)
                        buffer.append(
                            [src_input, src_segment, src_padding_mask])
                        total_qids.append(cur_qids)
                        total_labels.append(int(click_label))
                    except:
                        pass

                if len(buffer) >= self.buffer_size:  # we use 300,000 click records for test
                    break

        return buffer, total_qids, total_labels

    def load_data_feature(self, fpath, feature_path):
        if self.load_feature:
            print('load features', feature_path)
            annotate_features = np.loadtxt(feature_path)
            # if config.vote:
            #     res = vote(annotate_features)
            #     # === min-max
            #     annotate_features = (annotate_features - annotate_features.min(axis=0)) / \
            #         (annotate_features.max(axis=0) - annotate_features.min(axis=0))
            #     res = np.expand_dims(res, axis=1)
            #     annotate_features = np.concatenate(
            #         [annotate_features, res], axis=-1)
        print('load data from ', fpath)
        total_qids = []
        buffer = []
        total_labels = []
        total_freqs = []
        idx = 0
        for line in tqdm(open(fpath, 'rb')):
            line_list = line.strip(b'\n').split(b'\t')
            qid, query, title, content, label, freq = line_list
            if 0 <= int(freq) <= 2:  # high freq
                freq = 0
            elif 3 <= int(freq) <= 6:  # mid freq
                freq = 1
            elif 7 <= int(freq):  # tail
                freq = 2
            total_qids.append(int(qid))
            total_labels.append(int(label))
            total_freqs.append(freq)
            src_input, src_segment, src_padding_mask = process_data(
                query, title, content, self.max_seq_len)
            if self.load_feature:
                # ===test
                q_len = (len(query.split(b'\x01'))-9.505050054514818)/4.961076160898782
                # 替换掉query_len
                # q_len = res[idx]
                new_feat = np.append(annotate_features[idx], q_len)

                buffer.append([src_input, src_segment, src_padding_mask,
                               paddle.to_tensor(new_feat, dtype=paddle.float32)])
            else:
                buffer.append([src_input, src_segment, src_padding_mask])
            idx += 1

        return buffer, total_qids, total_labels, total_freqs


class Train_listDataset(IterableDataset):
    def __init__(self, directory_path, buffer_size=100000, max_seq_len=128, load_feature=0,
                 num_candidates=10, negative_num=0, query_num=10, train_feature_path='', mode=None, feature_pairs=None, train_freq_path=''):
        self.directory_path = directory_path
        self.buffer_size = buffer_size
        self.files = os.listdir(self.directory_path)
        random.shuffle(self.files)
        self.cur_query = "#"
        self.max_seq_len = max_seq_len
        self.load_feature = load_feature
        self.num_candidates = num_candidates
        self.negative_num = negative_num
        self.query_num = query_num
        # list[pos+neg] replace_list[最后一次点击后的random替换为负样本] pair
        self.mode = mode
        self.feature_pairs = feature_pairs
        # calculate statistics in total data_files in advance
        # {q-word:df}
        self.doc_df, self.title_df, self.content_df = {}, {}, {}
        # {word: cnt}
        self.doc_cnt, self.title_cnt, self.content_cnt = {}, {}, {}
        # avg len
        self.total_doc, self.docs_len, self.titles_len, self.contents_len = 0, 0, 0, 0
        # collection tokens
        self.qdoc_tokens, self.qtitle_tokens, self.qcontent_tokens = [], [], []
        self.train_feature_path = train_feature_path
        self.feature_id = type_idx(config.feature_type)
        self.train_freq_path = train_freq_path
    # --------------- statistics and generate features----------------  #

    def cal_df(self, query, title, content, doc):
        for word in query:
            if word in title:
                try:
                    self.title_df[word] += 1
                except:
                    self.title_df[word] = 1
            if word in content:
                try:
                    self.content_df[word] += 1
                except:
                    self.content_df[word] = 1
            if word in doc:
                try:
                    self.doc_df[word] += 1
                except:
                    self.doc_df[word] = 1

    def cal_total_feature(self, query, title, content, mean_std_file=config.mean_std_file):

        query = query.split(b'\x01')
        title = title.split(b'\x01')
        content = content.split(b'\x01')
        doc = title+content

        features = []

        features += cal_item_feature(query, title,
                                     self.title_df, self.qtitle_tokens, self.title_cnt, self.titles_len, self.total_doc)
        features += cal_item_feature(query, content, self.content_df,
                                     self.qcontent_tokens, self.content_cnt, self.contents_len, self.total_doc)
        features += cal_item_feature(query, doc, self.doc_df,
                                     self.qdoc_tokens, self.doc_cnt, self.docs_len, self.total_doc)
        features.append(len(query))
        # normalize features with Gaussion
        features = np.array(features)
        m, s = np.loadtxt(mean_std_file)
        # m = 0.0
        # s = 1.0
        return (features-m)/s

    def calculate_statistics(self, data_path):
        with gzip.open(data_path, 'rb') as f:
            for line in tqdm(f.readlines()):
                line_list = line.strip(b'\n').split(b'\t')
                if len(line_list) == 3:
                    query = line_list[1]
                    query = query.split(b'\x01')
                else:
                    self.total_doc += 1
                    title, content = line_list[2], line_list[3]
                    title = title.split(b'\x01')
                    content = content.split(b'\x01')
                    doc = title+content
                    self.docs_len += len(doc)
                    self.titles_len += len(title)
                    self.contents_len += len(content)
                    self.cal_df(query, title, content, doc)
                    cal_cnt(doc, self.doc_cnt)
                    cal_cnt(title, self.title_cnt)
                    cal_cnt(content, self.content_cnt)
        # avg_len
        self.docs_len /= self.total_doc
        self.titles_len /= self.total_doc
        self.contents_len /= self.total_doc
        self.qdoc_tokens, self.qtitle_tokens, self.qcontent_tokens = list(
            self.doc_df.keys()), list(self.title_df.keys()), list(self.content_df.keys())

    def gengerate_data(self, tmp_qid2dict):
        all_qids = list(tmp_qid2dict.keys())
        random.shuffle(all_qids)
        buffer = []
        for qid in tmp_qid2dict.keys():
            query_text = tmp_qid2dict[qid]['query_text']
            query_text_reformu = tmp_qid2dict[qid]['query_text_reformu']
            q_freq_ = tmp_qid2dict[qid]['q_freq']
            if len(tmp_qid2dict[qid]['doc_info']) < self.num_candidates:
                continue
            pos_doc_list = tmp_qid2dict[qid]['doc_info'][:self.num_candidates]
            # 全click为0的跳过&find last_click_pos
            click_cnt, last_click, idx = 0, 0, 0
            click_ids, skip_ids, features = [], [], []
            for pos_doc in pos_doc_list:
                if self.mode == 'pair':
                    features.append(pos_doc['feature'][self.feature_id])
                if pos_doc['click_label'] == 1:
                    click_cnt += 1
                    last_click = idx
                    click_ids.append(idx)
                else:
                    skip_ids.append(idx)
                idx += 1

            if click_cnt == 0:
                continue
            negative_num = self.negative_num
            # 调整negative_num
            if self.mode == 'replace_list':
                negative_num += self.num_candidates-last_click-1

            if negative_num != 0:
                neg_list = []  # all neg_list
                for neg_qid in all_qids:
                    if tmp_qid2dict[neg_qid]['query_text'] == query_text:
                        continue
                    neg_list.extend(
                        tmp_qid2dict[neg_qid]['doc_info'])
                    # 存储够了就提前退出
                    if len(neg_list) >= 2 * negative_num:
                        break
                neg_ids = [i for i in range(len(neg_list))]
                if len(neg_ids) == 0:  # 没有随机负样本
                    continue
                # 如果负样本数不足，可以重复采样
                replacement = False if len(
                    neg_ids) > negative_num else True
                neg_ids = np.random.choice(
                    neg_ids, negative_num, replace=replacement)
                neg_doc_list = [neg_list[neg_idx]
                                for neg_idx in neg_ids]
            else:
                neg_doc_list = []

            if self.mode == 'list' or self.mode == 'replace_list':
                idx = 0
                for doc_detail in pos_doc_list:
                    # 最后一次点击之后的click=0的替换为random neg
                    if self.mode == 'replace_list' and idx == last_click+1:
                        break
                    src_input, src_segment, src_padding_mask = process_data(
                        query_text, doc_detail['title'], doc_detail['content'], self.max_seq_len)
                    if config.combine or config.change_label != 'no':
                        feature = doc_detail['feature']
                        buffer.append([src_input, src_segment, src_padding_mask, doc_detail['click_label'],
                                       feature, q_freq_])
                    else:
                        buffer.append(
                            [src_input, src_segment, src_padding_mask, doc_detail['click_label'], q_freq_])
                    idx += 1
                for doc_detail in neg_doc_list:
                    src_input, src_segment, src_padding_mask = process_data(
                        query_text, doc_detail['title'], doc_detail['content'], self.max_seq_len)
                    if config.combine or config.change_label != 'no':
                        feature = self.cal_total_feature(
                            query_text, doc_detail['title'], doc_detail['content'])
                        if config.vote:
                            # 替换原本的feature
                            # ===test
                            # feature[type_idx('query_len')
                            #         ] = vote_for_neg(feature)
                            # feature = np.append(feature, 0)
                            feature = np.append(feature, vote_for_neg(feature))
                        feature = paddle.to_tensor(
                            feature, dtype=paddle.float32)
                        buffer.append([src_input, src_segment, src_padding_mask, 0.0,
                                       feature, q_freq_])
                    else:
                        buffer.append(
                            [src_input, src_segment, src_padding_mask, 0.0, q_freq_])

            elif self.mode == 'pair':
                # 不涉及负样本的pair
                pos_ids = []
                if len(skip_ids) != 0:
                    click_id, skip_id = np.random.choice(
                        click_ids, 1)[0], np.random.choice(skip_ids, 1)[0]
                    # [click=1,click=0]
                    pos_ids.extend([click_id, skip_id])
                feature_ids = sorted(
                    range(self.num_candidates), key=lambda i: features[i], reverse=True)
                # feature pairs
                for i in self.feature_pairs:
                    pos_ids.append(feature_ids[i-1])
                label = 1
                for idx in pos_ids:
                    doc_detail = pos_doc_list[idx]
                    src_input, src_segment, src_padding_mask = process_data(
                        query_text, doc_detail['title'], doc_detail['content'], self.max_seq_len)
                    if config.combine or config.change_label != 'no':
                        feature = doc_detail['feature']
                        buffer.append([src_input, src_segment, src_padding_mask, label,
                                       feature, q_freq_])
                    else:
                        buffer.append(
                            [src_input, src_segment, src_padding_mask, label, q_freq_])
                    label = 1-label

                # 涉及负样本的pair
                pos_ids = []
                if len(skip_ids) != 0:
                    skip_id = np.random.choice(skip_ids, 1)[0]
                    pos_ids.append(skip_id)
                pos_ids.extend(
                    [np.random.choice(click_ids, 1)[0], feature_ids[-1]])
                for i, j in zip(pos_ids, neg_ids[:len(pos_ids)]):
                    pos_doc, neg_doc = pos_doc_list[i], neg_list[j]
                    src_input, src_segment, src_padding_mask = process_data(
                        query_text, pos_doc['title'], pos_doc['content'], self.max_seq_len)
                    if config.combine or config.change_label != 'no':
                        feature = pos_doc['feature']
                        buffer.append([src_input, src_segment, src_padding_mask, 1,
                                       feature, q_freq_])
                    else:
                        buffer.append(
                            [src_input, src_segment, src_padding_mask, 1, q_freq_])
                    src_input, src_segment, src_padding_mask = process_data(
                        query_text, neg_doc['title'], neg_doc['content'], self.max_seq_len)
                    if config.combine or config.change_label != 'no':
                        feature = self.cal_total_feature(
                            query_text, doc_detail['title'], doc_detail['content'])
                        feature = paddle.to_tensor(
                            feature, dtype=paddle.float32)
                        buffer.append([src_input, src_segment, src_padding_mask, 0,
                                       feature, q_freq_])
                    else:
                        buffer.append(
                            [src_input, src_segment, src_padding_mask, 0, q_freq_])

        return buffer

    def __iter__(self):
        print('load train query freq: ', self.train_freq_path)
        train_freq = np.loadtxt(self.train_freq_path)
        if self.load_feature:
            print('load train features: ', self.train_feature_path)
            train_features = np.loadtxt(self.train_feature_path)

            if config.vote:
                # === min-max
                ori_res = vote(train_features)
                # 也对 fetaure 进行归一化
                # train_features = (train_features - train_features.min(axis=0)) / \
                #     (train_features.max(axis=0) - train_features.min(axis=0))
                # 替换掉原本的feature
                # train_features[:, type_idx('query_len')] = res
                res = np.expand_dims(ori_res, axis=1)
                train_features = np.concatenate([train_features, res], axis=-1)
        for file in self.files:
            if file[-3:] != '.gz' or file == 'part-00000.gz':  # part-00000.gz is for evaluation
                continue
            if config.combine or config.change_label != 'no':
                print('calculate file', file)
                self.calculate_statistics(
                    os.path.join(self.directory_path, file))

        for file in self.files:
            print('load file', file)
            if file[-3:] != '.gz' or file == 'part-00000.gz':  # part-00000.gz is for evaluation
                continue
            with gzip.open(os.path.join(self.directory_path, file), 'rb') as f:
                idx = 0
                tmp_num = 0  # 临时存放
                tmp_qid2dict = {}
                buffer = []
                query_num_ana = -1  
                for line in f.readlines():
                    line_list = line.strip(b'\n').split(b'\t')
                    if len(line_list) == 3:  # new query
                        query_num_ana += 1  
                        if self.cur_query != '#':
                            tmp_num += 1  # 目前的query数量

                        if tmp_num >= self.query_num:  # 根据batch内的query数量
                            buffer.extend(self.gengerate_data(
                                tmp_qid2dict))  # 生成样本
                            tmp_num = 0  # 重新设置为0
                            tmp_qid2dict = {}

                        if len(buffer) >= self.buffer_size:
                            for record in buffer:
                                yield record
                            buffer.clear()

                        qid = line_list[0]
                        query_text = line_list[1]
                        query_text_reformu = line_list[2]
                        if qid not in tmp_qid2dict:
                            tmp_qid2dict[qid] = {}
                            tmp_qid2dict[qid]['query_text'] = query_text
                            tmp_qid2dict[qid]['query_text_reformu'] = query_text_reformu
                            tmp_qid2dict[qid]['doc_info'] = []
                            tmp_qid2dict[qid]['q_freq'] = train_freq[query_num_ana]
                        self.cur_query = qid

                    elif len(line_list) > 6:  # urls
                        position, title, content, click_label = line_list[
                            0], line_list[2], line_list[3], line_list[5]
                        success = True
                        if len(title) == 0 or len(content) == 0:
                            success = False
                        if success:  # 对于title为0 或者content为0的pass掉
                            if self.load_feature:
                                q_len = (len(query_text.split(b'\x01')) -
                                         9.505050054514818)/4.961076160898782
                                # ===test
                                # q_len = ori_res[idx]
                                tmp_qid2dict[self.cur_query]['doc_info'].append({
                                    'title': title,
                                    'content': content,
                                    'click_label': float(click_label),
                                    'feature': paddle.to_tensor(np.insert(train_features[idx], [24], q_len), dtype=paddle.float32),
                                })
                            else:
                                tmp_qid2dict[self.cur_query]['doc_info'].append({
                                    'title': title,
                                    'content': content,
                                    'click_label': float(click_label),
                                })
                        idx += 1

                buffer.extend(self.gengerate_data(tmp_qid2dict))  # 生成样本
                for record in buffer:
                    yield record


def build_feed_dict(data_batch, load_feature=0, size=10, mode=None):
    if load_feature:  # 多了features
        if len(data_batch) == 6:  # for training
            src, src_segment, src_padding_mask, label, features, q_freq = data_batch
        elif len(data_batch) == 4:  # for validation
            src, src_segment, src_padding_mask, features = data_batch
        else:
            raise KeyError

        feed_dict = {
            'src': src,
            'src_segment': src_segment,
            'src_padding_mask': src_padding_mask,
            'features': features
        }
        # 不需要label default=>10
        if mode == 'pair':
            if len(data_batch) == 6:
                feed_dict['q_freq'] = q_freq
            return feed_dict
        if len(data_batch) == 6:
            click_label = label.numpy().reshape(-1, size).T
            for i in range(size):
                feed_dict['label' + str(i)] = click_label[i]
            feed_dict['q_freq'] = q_freq
        return feed_dict
    if len(data_batch) == 5:  # for training
        src, src_segment, src_padding_mask, label, q_freq = data_batch
    elif len(data_batch) == 3:  # for validation
        src, src_segment, src_padding_mask = data_batch
    else:
        raise KeyError

    feed_dict = {
        'src': src,
        'src_segment': src_segment,
        'src_padding_mask': src_padding_mask,
    }
    if len(data_batch) == 5:
        label = label.numpy().reshape(-1, size).T
        for i in range(size):
            feed_dict['label'+str(i)] = label[i]
        feed_dict['q_freq'] = q_freq

    return feed_dict
