import numpy as np
import gzip
from tqdm import tqdm
import paddle.nn.functional as F
import paddle


def norm():
    train_feature_path = '/ossfs/workspace/wsdm_cup/data/new_train/features/part-00001.txt'
    train_features = np.loadtxt(train_feature_path)
    m = train_features.mean(axis=0)
    s = train_features.std(axis=0)
    s[s == 0] = 1
    np.savetxt('/ossfs/workspace/wsdm_cup/data/new_train/features/mean_std.txt',
               np.array([m, s]), fmt='%f', delimiter=' ')


def check_click_p():
    fpath = '/ossfs/workspace/wsdm_cup/data/train_data/part-00001.gz'
    train_freq = np.loadtxt(
        '/ossfs/workspace/wsdm_cup/data/new_train/query_frequency.txt')
    all_query_cnt = 0
    bad_query = 0
    now_query_label_list = []
    ten_pos_cnt = {}
    all_click_num = 0
    have_multi_clicl_query = 0
    query_num_ana = -1  
    with gzip.open(fpath, 'rb') as f:
        for line in tqdm(f.readlines()):
            line_list = line.strip(b'\n').split(b'\t')
            if len(line_list) == 3:  # new query
                query_num_ana += 1  
                query_freq = train_freq[query_num_ana]
                if query_freq > 1:  # 只用高频的
                    now_query_label_list = []
                    continue
                if len(now_query_label_list) < 10:
                    now_query_label_list = []
                    bad_query += 1
                    continue
                click_num = 0
                for idx, label_ in enumerate(now_query_label_list):
                    if idx not in ten_pos_cnt:
                        ten_pos_cnt[idx] = {
                            1: 0,
                            0: 0,
                            'all': 0
                        }
                    ten_pos_cnt[idx][label_] += 1
                    ten_pos_cnt[idx]['all'] += 1
                    if label_ == 1:
                        all_click_num += 1
                        click_num += 1
                if click_num != 0:
                    have_multi_clicl_query += 1

                all_query_cnt += 1
                now_query_label_list = []
            elif len(line_list) > 6:  # urls
                position, title, content, click_label = line_list[
                    0], line_list[2], line_list[3], line_list[5]
                now_query_label_list.append(int(click_label))
    print(ten_pos_cnt)
    print(bad_query)
    print(all_query_cnt)
    print(have_multi_clicl_query, all_query_cnt,
          have_multi_clicl_query/all_query_cnt)

    for pos in ten_pos_cnt.keys():
        print('pos:', pos, 'click_prob:',
              ten_pos_cnt[pos][1]/ten_pos_cnt[pos]['all'])
    for pos in ten_pos_cnt.keys():
        print('pos:', pos, 'click_prob:', ten_pos_cnt[pos][1]/all_click_num)


def check_human():
    fpath = '/ossfs/workspace/wsdm_cup/data/new_train/finetine_train.txt'
    qrels = {}
    for line in tqdm(open(fpath, 'rb')):
        line_list = line.strip(b'\n').split(b'\t')
        qid, query, title, content, label, freq = line_list
        if qid not in qrels:
            qrels[qid] = []
        qrels[qid].append(label)
    all_cnt = []
    for qid in qrels.keys():
        all_cnt.append(len(qrels[qid]))

    all_cnt = np.array(all_cnt)
    print(len(all_cnt))
    print(np.max(all_cnt), np.min(all_cnt), np.mean(all_cnt))


def type_idx(type):
    mapping = {}
    features = []
    for item in ['title', 'content', 'title_content']:
        for feature in ['tf', 'idf', 'tfidf', 'len', 'bm25', 'JM', 'DIR', 'ABS']:
            features.append('{}_{}'.format(feature, item))
    for idx, feature in enumerate(features):
        mapping[feature] = idx
    return mapping[type]


def vote(feature):

    def norm_feature(feat):
        feat_min = feat.min()
        feat_max = feat.max()
        norm_feat = (feat - feat_min) / (feat_max - feat_min)
        return norm_feat

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


def test_vote_feature():
    train_feature_path = '/ossfs/workspace/wsdm_cup/data/new_train/features/norm_part-00001.txt'
    train_features = np.loadtxt(train_feature_path)
    print(train_features.shape)
    """
    min: 24 个
    [ -1.040408,  -1.88205 ,  -1.08962 ,  -1.425753,  -0.569995,
        -7.176515,  -1.362688, -15.078766,  -0.253144,  -1.890123,
        -0.279212,  -0.317139,  -0.140559,  -8.424216,  -2.198761,
       -35.757577,  -1.002119,  -1.888678,  -1.071076,  -0.739215,
        -0.600482,  -7.331542,  -1.986095, -17.550349]
    max: 
    [ 62.025213,  16.612423,  70.137452,  41.460367, 111.194629,
        12.159794,  31.199346,  14.509522,  85.52469 ,  14.898351,
        95.922703, 122.0741  , 501.753289,  15.33013 ,  44.701508,
        27.568491,  69.030029,  16.446504,  79.255638, 112.816226,
       148.814068,  11.006035,  31.93112 ,  12.540787]
    """
    new_train_features = (train_features - train_features.min(axis=0)) / \
        (train_features.max(axis=0) - train_features.min(axis=0))
    res = vote(train_features)
    import pdb;pdb.set_trace()
    res = np.expand_dims(res, axis=1)
    train_features = np.concatenate(
        [train_features, res], axis=-1)  # 25个feature

    fpath = '/ossfs/workspace/wsdm_cup/data/train_data/part-00001.gz'
    click_label_list = []
    feature_list = []

    with gzip.open(fpath, 'rb') as f:
        idx = -1
        for line in tqdm(f.readlines()):
            line_list = line.strip(b'\n').split(b'\t')
            if len(line_list) == 3:  # new query
                click_num = 0
                if len(click_label_list) < 10:
                    click_label_list = []
                    feature_list = []
                    continue
                for idx, label_ in enumerate(click_label_list[:10]):
                    if label_ == 1:
                        click_num += 1
                if click_num == 0:
                    click_label_list = []
                    feature_list = []
                    continue
                else:
                    # click_label_list=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
                    # vote: [2.5163626736071345, 2.266514273292788, 2.548663546679607, 2.5311298982796373, 2.5085705737930746, 2.5171902262463104, 2.4596189195039297, 2.4657179631159614, 2.4893108397700017, 2.464101467691373]
                    # jmtitle: [0.19153, -0.971846, 0.308093, 0.342512, 0.242504, 0.275406, 0.082035, 0.108679, 0.184632, 0.101674]
                    import pdb
                    pdb.set_trace()
                click_label_list = []
                feature_list = []
            elif len(line_list) > 6:  # urls
                idx += 1
                position, title, content, click_label = line_list[
                    0], line_list[2], line_list[3], line_list[5]
                click_label_list.append(int(click_label))
                feature_list.append(train_features[idx])


norm()