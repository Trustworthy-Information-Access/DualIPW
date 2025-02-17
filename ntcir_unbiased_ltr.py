from baseline_model.utils.sys_tools import find_class
import torch
torch.set_printoptions(10)
import numpy as np
import warnings
import sys
from metrics import *
from args import config
# 数据集读入
from ntcir_dataset import *
import time
import os
from collections import OrderedDict
import random
from torch.utils.data import DataLoader

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
warnings.filterwarnings('ignore')
print(config)
exp_settings = config.exp_settings


def save_dict(save_path, dict_need):
    import json
    with open(save_path, 'w', encoding='utf-8') as f_out:
        json.dump(dict_need, f_out, ensure_ascii=False, indent=2)


def eval(key, save=True,save_m=False):
    global best_eval_score, best_eval_step
    if type(key) == str or key % config.eval_step == 0 or key == -1:
        # ------------   evaluate on annotated data -------------- #

        # valid
        total_scores = []
        with torch.no_grad():
            for test_data_batch in vaild_annotate_loader:
                feed_input = build_feed_dict(
                    test_data_batch)
                # print(feed_input)
                score = model.get_scores(feed_input)
                score = score.cpu().detach().numpy().tolist()
                total_scores += score
        # print(len(total_scores),len(vaild_annotate_dataset.total_labels))
        # print(torch.Tensor(total_scores).shape)
        result_dict_ann = evaluate_all_metric(
            qid_list=vaild_annotate_dataset.total_qids,
            label_list=vaild_annotate_dataset.total_labels,
            score_list=total_scores,
            file_dir=f'{config.output_name}/',
            save=save_m
        )
        valid_dict[key] = {
            'ndcg@10': result_dict_ann["all_ndcg@10"],
            'pnr': result_dict_ann["pnr"],
        }
        if method_str.startswith('UPE') and key != -1:
            valid_dict[key]['weights'] = weights
        print(
            f'{key}th step valid annotate | '
            f'@10 ndcg: all {result_dict_ann["all_ndcg@10"]:.6f} | '
            f'err {result_dict_ann["all_err@10"]:.6f} | '
            f'pnr {result_dict_ann["pnr"]:.6f}'
        )
        if type(key)==str and 'valid' in key:
            print(
                f'@10 ndcg: all {result_dict_ann["all_ndcg@10"]:.6f} | '
                f'@5 ndcg: all {result_dict_ann["all_ndcg@5"]:.6f} | '
                f'@3 ndcg: all {result_dict_ann["all_ndcg@3"]:.6f} | '
                f'@1 ndcg: all {result_dict_ann["all_ndcg@1"]:.6f} | '
                f'@10 dcg: all {result_dict_ann["all_dcg@10"]:.6f} | '
                f'@5 dcg: all {result_dict_ann["all_dcg@5"]:.6f} | '
                f'@3 dcg: all {result_dict_ann["all_dcg@3"]:.6f} | '
                f'@1 dcg: all {result_dict_ann["all_dcg@1"]:.6f} | '
                f'@10 err: all {result_dict_ann["all_err@10"]:.6f} | '
                f'@5 err: all {result_dict_ann["all_err@5"]:.6f} | '
                f'@3 err: all {result_dict_ann["all_err@3"]:.6f} | '
                f'@1 err: all {result_dict_ann["all_err@1"]:.6f} | '
            )

        eval_score = result_dict_ann["all_ndcg@10"]
        if save and (best_eval_score is None or eval_score > best_eval_score):
            best_eval_score = eval_score
            best_eval_step = key
            
            torch.save(model.state_dict(),
                       config.output_name + f'/best_model.model')
            print('new top test score {} at step-{}, saving weights'.format(
                best_eval_score, best_eval_step))
            valid_dict[f'best_step_{best_eval_step}'] = best_eval_score
        save_dict(config.output_name + '/valid_dict.json', valid_dict)
        save_dict(config.output_name + '/loss.json', loss_dict)


method_str = exp_settings['method_name']
ranking_method = config.ranking_method
total_methods = []
for m in ['DLA', 'NaiveAlgorithm', 'UPE','PRS','Gradrev','Drop','DualIPW']:
    total_methods.append(f'{m}_DNN')
if method_str not in total_methods:
    print(f"please choose a method in {[m for m in total_methods]}")
    sys.exit()

if not os.path.exists(config.output_name):  # 判断所在目录下是否有该文件名的文件夹
    os.makedirs(config.output_name)

start = time.time()

valid_str = 'revalid'
vaild_annotate_dataset = TestDataset(valid_str,config.eval_type)
vaild_annotate_loader = DataLoader(
    vaild_annotate_dataset, batch_size=config.eval_batch_size)

# loss NDCG@10
valid_dict = {}
loss_dict = {}
best_eval_step = 0
best_eval_score = None

idx = -1


total_scores = []

# ranking model
ranking_model = find_class('baseline_model.ranking_model.'+ranking_method)(
    config.projection, config.rank_feature_size)
model = find_class('baseline_model.learning_algorithm.' +
                   method_str)(exp_settings=exp_settings, ranking_model=ranking_model)

# step -1: 模型初始化的效果
if config.iter:
 
    print(f'load {config.init_model}')
    init_model = torch.load(config.init_model)
    model.load_state_dict(init_model)
    eval('init')

idx = -1
for epoch in range(config.epoch):

    train_dataset = TrainDataset(
        buffer_size=config.buffer_size, epoch_idx=epoch, num_candidates=config.num_candidates)

    # print(config.train_batch_size)
    train_data_loader = DataLoader(
        train_dataset, batch_size=config.train_batch_size)
    # sum_q=0
    for train_batch in train_data_loader:
        
        eval(idx)
        idx += 1
        if method_str.startswith('UPE'):
            loss, weights = model.train(build_feed_dict(
                train_batch))
            weights = list(map(lambda x: float(x), weights.cpu().numpy()))
        else:
            loss = model.train(build_feed_dict(
                train_batch))

        if idx % config.log_interval == 0:
            loss_dict[idx] = loss
            print(f'{idx:5d}th step | loss {loss:5.6f}')

    # ------------   evaluate on annotated data -------------- #

    # valid
    eval(f'final_epoch{epoch}')
    torch.save(model.state_dict(),
            config.output_name + f'/final{epoch}_model.model')
    print(f'total run time:{time.time()-start}')
    valid_dict['run_time'] = time.time()-start
    save_dict(config.output_name + '/valid_dict.json', valid_dict)

# valid eval
model.load_state_dict(torch.load(config.output_name + f'/best_model.model'))

test_str='validQsplit'
vaild_annotate_dataset = TestDataset(valid_str, test_str)
vaild_annotate_loader = DataLoader(
    vaild_annotate_dataset, batch_size=config.eval_batch_size)
eval(f'best valid epoch{epoch}', save=False,save_m=True)
print(config.output_name)

