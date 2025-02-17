import numpy as np
import warnings
import sys
from args import config
from ntcir_dataset import *
import time
import os
import random
import torch
from torch.utils.data import DataLoader

import lightgbm

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
warnings.filterwarnings('ignore')


def ret_data(data_type, limit=None):
    dataset = TestDataset('', data_type, 'click', limit=limit)
    data_loader = DataLoader(
        dataset, batch_size=config.train_batch_size)
    # sum_q=0
    total_x, total_y = [], []
    total_qs = dataset.total_qs
    groups = [10]*(total_qs)
    # int or float? cls or reg?
    for i in range(total_qs):
        total_y += list(map(lambda k: 10-k, list(range(1, 11))))

    for batch in data_loader:
        cur_input = build_feed_dict(
            batch, size=config.num_candidates)
        features = cur_input['feat_input'].numpy().tolist()

        total_x += features
    print(total_qs, len(total_x), len(total_y), sum(groups))
    return total_x, total_y, groups, total_qs, dataset.total_dids


def cal_dcg(pred=None, ordered_labels=None):
    if pred is not None:
        ordered_labels = np.array(ordered_labels)[np.argsort(pred)[::-1]]
    return np.sum((2**ordered_labels-1)/np.log2(np.arange(ordered_labels.shape[0])+2))


def cal_ndcg(pred, labels):
    ordered_labels = np.array(labels)[np.argsort(pred)[::-1]]
    dcg = cal_dcg(ordered_labels=ordered_labels)
    idcg = cal_dcg(ordered_labels=np.sort(labels)[::-1])
    return dcg/idcg


def train():
    start = time.time()
    lambdamart_ranker = lightgbm.LGBMRanker(
        boosting_type="gbdt",
        objective="lambdarank",
        n_estimators=config.n_trees,
        importance_type="gain",
        metric=None,
        num_leaves=config.n_leaves,
        learning_rate=config.lgb_lr,
        max_depth=-1,
        random_state=config.seed
    )

    total_x, total_y, groups, _, _ = ret_data('train', limit=290000)
    print('training')
    lambdamart_ranker.fit(total_x, total_y, group=groups)
    lambdamart_ranker.booster_.save_model(f'{config.output_name}/params.txt')
    mid_t = time.time()
    print('training time', mid_t-start)
    print('training time', mid_t-start, file=res_f)


def test():
    mid_t = time.time()
    ranker = lightgbm.Booster(model_file=f'{config.output_name}/params.txt')
    print('validating')
    total_x, total_y, groups, total_qs, _ = ret_data('valid')
    # [nsamples]
    pred = ranker.predict(total_x)
    metric_fns = {'ndcg@10': cal_ndcg, 'dcg@10': cal_dcg}
    metrics = {'ndcg@10': [], 'dcg@10': []}
    for i in range(total_qs):
        for m in ['ndcg@10', 'dcg@10']:
            metrics[m].append(metric_fns[m](
                pred[i*10:(i+1)*10], total_y[i*10:(i+1)*10]))
    print('validating time', time.time()-mid_t)
    print('validating time', time.time()-mid_t, file=res_f)

    for m in ['ndcg@10', 'dcg@10']:
        print(m, np.mean(metrics[m]))
        print(m, np.mean(metrics[m]), file=res_f)


def gen():
    ranker = lightgbm.Booster(model_file=f'{config.output_name}/params.txt')
    print('gen initial scores')
    total_x, total_y, groups, total_qs, total_dids = ret_data('train')
    # [nsamples]
    pred = ranker.predict(total_x)
    print(len(pred), len(total_dids))
    with open(config.data_file, 'w') as f:
        for did, p in zip(total_dids, pred):
            f.write(f'{did} {p}\n')


if __name__ == '__main__':

    if not os.path.exists(config.output_name):  # 判断所在目录下是否有该文件名的文件夹
        os.makedirs(config.output_name)

    res_f = open(f'{config.output_name}/valid.txt', 'a+')
    if config.test_only:
        test()
    elif config.gen_only:
        gen()
    else:
        train()
        test()
    res_f.close()
