from args import config
from baseline_model.utils.sys_tools import find_class
import torch
import torch.nn as nn
import numpy as np
import warnings
import sys
from metrics import *
# 数据集载入
from ntcir_dataset import *
from args import config
import os
import random
from torch.utils.data import DataLoader

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
warnings.filterwarnings('ignore')
print(config)
exp_settings = config.exp_settings

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

# loss NDCG@10
valid_dict = {}

idx = -1


total_scores = []

# ranking model
ranking_model = find_class('baseline_model.ranking_model.'+ranking_method)(
    config.projection, config.rank_feature_size)
model = find_class('baseline_model.learning_algorithm.' +
                    method_str)(exp_settings=exp_settings, ranking_model=ranking_model)
model.load_state_dict(torch.load(config.init_model))

valid_str = 'revalid'

vaild_annotate_dataset = TestDataset(valid_str, config.eval_type)
vaild_annotate_loader = DataLoader(
    vaild_annotate_dataset, batch_size=config.eval_batch_size)


# evaluate
total_scores = []
with torch.no_grad():
    for test_data_batch in vaild_annotate_loader:
        feed_input = build_feed_dict(
            test_data_batch)
        score = model.get_scores(feed_input)
        score = score.cpu().detach().numpy().tolist()
        total_scores += score
total_labels=vaild_annotate_dataset.total_labels
total_cnt=len(total_labels)

result_dict_ann = evaluate_all_metric(
    qid_list=vaild_annotate_dataset.total_qids,
    label_list=total_labels,
    score_list=total_scores,
    file_dir=config.init_model[:-16],
    # topN=10
)
print(f'top 10 docs'
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
