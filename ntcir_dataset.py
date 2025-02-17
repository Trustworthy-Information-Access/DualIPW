from typing import Iterator
from torch.utils.data import Dataset, IterableDataset
import torch
import os
from tqdm import tqdm
from args import config
import json
import numpy as np


data_path = config.data_path
gen_sim = config.gen_sim
tra_feat_size = config.tra_feat_size
combine = config.combine
total = config.total

def read_feat(eval_type, feat_type, large_emb, feat_total, combine_al):

    total_combine_al = combine_al
    score_path = f'{data_path}/{eval_type}_{feat_type}.feature'
    print(f'load score path:{score_path} size:{large_emb}')
    with open(score_path, 'r') as f:
        for line in tqdm(f.readlines()):
            line_list = line.strip('\n').split(' ')
            doc_id = line_list[0]
            # 是否加tradition feat
            if combine and combine_al != 1:
                feat_total[doc_id] = [0] * tra_feat_size
                total_combine_al = 1
            else:
                if doc_id not in feat_total:
                    feat_total[doc_id] = []

            if feat_type == 'score':
                feat_total[doc_id] += [float(line_list[1])]
            elif feat_type == 'pretrain':
                pre_size = len(feat_total[doc_id])
                feat_total[doc_id] += [0.0] * large_emb
                for cur_feat in line_list[1:]:
                    feat_idx, feat_val = cur_feat.split(':')
                    feat_total[doc_id][int(feat_idx) +
                                       pre_size] = float(feat_val)

    return total_combine_al, feat_total


class TestDataset(Dataset):

    def __init__(self, eval_type, data_type='',data_src='annotation',limit=None):

        if data_src=='annotation':
            if 'valid' in eval_type:
                self.buffer, self.total_qids, self.total_labels, self.total_dids, self.total_freqs = self.load_data(
                    eval_type, data_type)
            elif 'test' in eval_type:
                self.buffer, self.total_qids, self.total_dids = self.load_data(
                    eval_type)
        else: # click
            self.buffer,self.total_qids,self.total_labels,self.total_dids,self.total_qs=self.load_click_data(data_type,limit=limit)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

    def load_click_data(self,data_type,limit=None):

        feat_total={}
        # 文档的特征文件读取
        # 13 + 1/768/769

        if gen_sim or total:
            combine_al, feat_total = read_feat(
                'train', 'score', 1, feat_total, 0)

        if combine:
            tradition_path = f'{data_path}/train_{config.tra_type}.feature'
            print(f'load tradition path {tradition_path}')

            with open(tradition_path, 'r') as f:
                for line in f.readlines():
                    line_list = line.strip('\n').split(' ')
                    for feat in line_list[1:-1]:
                        idx, val = feat.split(':')
                        feat_total[line_list[0]][int(idx)] = float(val)
        # 读取列表
        buffer, total_labels, total_qids, total_dids = [], [], [], []
        total_qs=0
        with open(f'{data_path}/click/{data_type}.init_list', 'r') as f:
            with open(f'{data_path}/click/{data_type}.{config.label_type}', 'r') as g:
                # valid和test only top10
                for init_list, labels in zip(f.readlines(), g.readlines()):

                    qid = init_list.strip('\n').split(':')[
                        0]#.split('_')[0]
                    init_lists, labels = init_list.strip('\n').split(':')[1].split(
                        ' ')[:config.num_candidates], labels.strip('\n').split(':')[1].split(' ')[:config.num_candidates]


                    labels = [int(label) for label in labels]

                    # 有点击
                    if config.click_num == 'total' and sum(labels) > 0 or config.click_num == 'single' and sum(
                            labels) == 1 or config.click_num == 'multi' and sum(labels) > 1:
                        
                            
                        # 单点击某位置为训练数据
                        if config.click_num=='single' and config.click_pos!=-1 and click_pos_t!=config.click_pos-1:
                            continue
                        
                        
                        if limit is not None and total_qs>=limit:
                            break

                        for doc_id, click in zip(init_lists, labels):
                            buffer.append(
                                [torch.Tensor(feat_total[doc_id])])
                            total_labels.append(int(click))
                            total_qids.append(qid)
                            total_dids.append(doc_id)
                        total_qs+=1
                        

        return buffer, total_qids, total_labels, total_dids,total_qs

    
    def load_data(self, init_eval_type, data_type=''):

        feat_total = {}

        q_freqs=json.load(open(f'{data_path}/valid_freq.json','r'))
        for qid,freq in q_freqs.items():
            if freq <=2:
                freq=0
            elif freq<7:
                freq=1
            else:
                freq=2
            q_freqs[qid]=freq

        eval_type = init_eval_type.split('_')[0]

        # 文档的特征文件读取
        # 13 + 1/768/769
        if gen_sim or total:
            combine_al, feat_total = read_feat(
                eval_type, 'score', 1, feat_total, 0)

        if combine:
            tradition_path = f'{data_path}/{eval_type}_{config.tra_type}.feature'

            print(f'load tradition path {tradition_path}')

            with open(tradition_path, 'r') as f:
                for line in f.readlines():
                    line_list = line.strip('\n').split(' ')
                    if not gen_sim:
                        feat_total[line_list[0]] = [0] * 24
                    for feat in line_list[1:-1]:
                        idx, val = feat.split(':')
                        feat_total[line_list[0]][int(idx)] = float(val)


        buffer, total_labels, total_qids, total_dids, total_freqs = [], [], [], [], []
        total_q=0
        if 'valid' in init_eval_type:
            print(data_type)
            with open(f'{data_path}/{data_type}.init_list', 'r') as f:
                with open(f'{data_path}/{data_type}.labels', 'r') as g:
                    for init_list, labels in zip(f.readlines(), g.readlines()):
                        total_q+=1
                        qid = init_list.strip('\n').split(':')[0]
                        # if init_list.strip('\n').split(':')[0]!=labels.strip('\n').split(':')[0]:
                        #     print('valid misalign')
                        init_list, labels = init_list.strip('\n').split(':')[1].split(
                            ' '), labels.strip('\n').split(':')[1].split(' ')
                        list_len = len(init_list)
                        
                        cur_cnt=0
                        for doc_id, label in zip(init_list, labels):

                            feat_len = len(feat_total[doc_id])
                            buffer.append(
                                [torch.Tensor(feat_total[doc_id])])
                            cur_cnt+=1
                            total_labels.append(int(label))
                            total_qids.append(qid)
                            total_dids.append(doc_id)
                            total_freqs.append(q_freqs[qid])
                        
            print(len(buffer),len(total_qids),len(labels),total_q)
            return buffer, total_qids, total_labels, total_dids, total_freqs
        elif 'test' in init_eval_type:

            with open(f'{data_path}/{init_eval_type}.init_list', 'r') as f:
                for init_list in f.readlines():
                    qid = init_list.strip('\n').split(':')[0]
                    init_list = init_list.strip('\n').split(':')[1].split(' ')
                    for doc_id in init_list:
                        buffer.append([torch.Tensor(feat_total[doc_id])])

                        total_qids.append(qid)
                        total_dids.append(doc_id)

            return buffer, total_qids, total_dids


class TrainDataset(IterableDataset):

    def __init__(self, buffer_size=100000, epoch_idx=1, num_candidates=10):
        self.num_candidates = num_candidates
        self.buffer_size = buffer_size
        self.epoch_idx = epoch_idx
        self.did_list = []
        # print(f'candidate: {self.num_candidates}')

    def __iter__(self):


        feat_total={}

        initial_feat={}
        with open(f'{data_path}/train.initial_scores','r') as f:
            for line in f.readlines():
                did,d_score=line.strip('\n').split(' ')
                initial_feat[did]=float(d_score)

        # 文档的特征文件读取
        # 13 + 1/768/769

        if gen_sim or total:
            combine_al, feat_total = read_feat(
                'train', 'score', 1, feat_total, 0)

        if combine:
            tradition_path = f'{data_path}/train_{config.tra_type}.feature'
            print(f'load tradition path {tradition_path}')

            with open(tradition_path, 'r') as f:
                for line in f.readlines():
                    line_list = line.strip('\n').split(' ')
                    for feat in line_list[1:-1]:
                        idx, val = feat.split(':')
                        feat_total[line_list[0]][int(idx)] = float(val)

        self.buffer = []
        total_qs,less=0,0
        # 先筛得到candidates>=10 + 有点击
        with open(f'{data_path}/click/train.init_list', 'r') as f:
            with open(f'{data_path}/click/train.{config.label_type}', 'r') as g:

                for init_list, labels in zip(
                        f.readlines(), g.readlines()):
                    # if init_list.strip('\n').split(':')[0]!=labels.strip('\n').split(':')[0]:
                    #         print('train misalign')
                    cur_qid = init_list.strip('\n').split(':')[
                        0].split('_')[0]
                    init_lists, labels = init_list.strip('\n').split(':')[1].split(
                        ' '), labels.strip('\n').split(':')[1].split(' ')
                    if len(init_lists)<self.num_candidates:
                        continue
      
                    labels = [int(label) for label in labels][:self.num_candidates]
                    

                    if sum(labels) == 1:
                        click_pos_t = labels.index(1)

                    # 有点击
                    if config.click_num == 'total' and sum(labels) > 0 or config.click_num == 'single' and sum(
                            labels) == 1 or config.click_num == 'multi' and sum(labels) > 1:

                        # 单点击某位置为训练数据
                        if config.click_num=='single' and config.click_pos!=-1:
                            if click_pos_t!=config.click_pos-1:
                                continue

                        click_times=1
                        if config.click_mode=='single':
                            click_times=sum(labels)
                        clicks_pos=[]
                        for idx,l in enumerate(labels):
                            if l==1:
                                clicks_pos.append(idx)
                        for i in range(click_times):
                            if config.click_mode=='single':
                                labels=[0]*self.num_candidates
                                labels[clicks_pos[i]]=1
                            for doc_id, click in zip(
                                    init_lists[:self.num_candidates], labels):
                                self.buffer.append(
                                    [torch.Tensor(feat_total[doc_id]), click,initial_feat[doc_id]])
                                self.did_list.append(doc_id)
                                

                        total_qs += 1

                    if len(self.buffer) > self.buffer_size:
                        for record in self.buffer:
                            yield record
                        self.buffer = []
        for record in self.buffer:
            yield record
        print(total_qs,less)


def build_feed_dict(data_batch, size=10, mode=None):

    feed_dict = {}
    if len(data_batch) == 2:  # eval
        feat_input, label = data_batch
    elif len(data_batch) == 3:
        feat_input, label, initial_scores = data_batch
        feed_dict['initial_scores']=initial_scores
        # print(len(initial_scores))
    elif len(data_batch) == 1:  # test
        feat_input = data_batch[0]
    else:
        raise KeyError

    feed_dict['feat_input'] = feat_input

    # 不需要label default=>10
    if mode == 'pair':
        return feed_dict

    if len(data_batch) != 1:

        click_label = label.numpy().reshape(-1, size)
        click_feat=feat_input.numpy().reshape(list(click_label.shape)+[-1])
        click_label=click_label.T
        click_feat=click_feat.transpose(1,0,2)
        # [seq_len bs x]
        for i in range(size):
            feed_dict['label' + str(i)] = click_label[i]
            feed_dict['docid_input'+str(i)]=click_feat[i]

    return feed_dict
