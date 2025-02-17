# -*- encoding: utf-8 -*-
'''
@Time    :   2022/06/10 15:51:44
@Author  :   Chu Xiaokai
@Contact :   xiaokaichu@gmail.com
'''
import numpy as np


def get_dcg(ordered_labels):
    return np.sum((2 ** ordered_labels - 1) / np.log2(np.arange(ordered_labels.shape[0]) + 2))


def get_idcg(complete_labels, max_len):
    return get_dcg(np.sort(complete_labels)[:-1 - max_len:-1])


def get_err_k(ranked_labels, K):
    err = 0.0
    R = (np.exp2(ranked_labels) - 1) / (2**4)
    for k in range(1, K+1):
        tmp = 1. / k
        for i in range(1, k):
            tmp *= (1 - R[i-1])
        tmp *= R[k-1]
        err += tmp
    return err


def calc_err(query_list, K=[1, 3, 5, 10], prefix='',file_dir=None,metrics_file=None):
    """ expected reciprocal rank """
    errs = [[], [], [], []]
    for item in query_list:
        pred, label = zip(*item)
        label = np.array(label)
        ranking = np.argsort(pred)[::-1]

        for i, k in enumerate(K):
            if len(pred) >= k:
                ranked_labels = label[ranking[:k]]
                this_err = get_err_k(ranked_labels, k)
                errs[i].append(this_err)

    if metrics_file is not None:
        for i,k in enumerate(K):
            metrics_file[f'err@{k}']=errs[i]
    
    if file_dir is not None:
        for i,k in enumerate(K):
            with open(f'{file_dir}err@{k}.txt','w') as f:
                for m in errs[i]:
                    f.write(f'{m}\n')
    
    return {prefix + '_err@'+str(k): np.mean(errs[i]) for i, k in enumerate(K)}


def calc_dcg(query_list, K=[1, 3, 5, 10], prefix='',file_dir=None,metrics_file=None):
    """ discounted cumulative gain """
    dcgs = [[], [], [], []]
    for item in query_list:
        pred, label = zip(*item)
        label = np.array(label)
        ranking = np.argsort(pred)[::-1]
        for i, k in enumerate(K):
            if len(pred) >= k:
                topk_rankings = ranking[:k]
            else:
                topk_rankings = ranking
            ordered_label = label[topk_rankings]
            dcgs[i].append(get_dcg(ordered_label))
    
    if metrics_file is not None:
        for i,k in enumerate(K):
            metrics_file[f'dcg@{k}']=dcgs[i]
    
    if file_dir is not None:
        for i,k in enumerate(K):
            with open(f'{file_dir}dcg@{k}.txt','w') as f:
                for m in dcgs[i]:
                    f.write(f'{m}\n')
    return {prefix + '_dcg@'+str(k): np.mean(dcgs[i]) for i, k in enumerate(K)}


def calc_ndcg(query_list, K=[1, 3, 5, 10], prefix='',file_dir=None,metrics_file=None):
    """  normalized discounted cumulative gain   """
    ndcgs = [[], [], [], []]
    for item in query_list:
        pred, label = zip(*item)
        label = np.array(label)
        ranking = np.argsort(pred)[::-1]

        for i, k in enumerate(K):
            if len(pred) >= k:
                dcg = get_dcg(label[ranking[:k]])
                idcg = get_idcg(label, max_len=k) + 10e-9
                ndcgs[i].append((dcg/idcg))
    
    if metrics_file is not None:
        for i,k in enumerate(K):
            metrics_file[f'ndcg@{k}']=ndcgs[i]
    
    if file_dir is not None:
        for i,k in enumerate(K):
            with open(f'{file_dir}ndcg@{k}.txt','w') as f:
                for m in ndcgs[i]:
                    f.write(f'{m}\n')
    return {prefix + '_ndcg@'+str(k): np.mean(ndcgs[i]) for i, k in enumerate(K)}


def calc_pnr(query_list):
    """ positive negative rate 
        = positive pairs / negative pairs
    """
    pos_pair = 0.0
    neg_pair = 10e-9
    fair_pair = 0
    for item in query_list:
        for i in range(len(item)):
            for j in range(i+1, len(item)):
                if (item[i][0] > item[j][0] and item[i][1] > item[j][1]) or \
                        (item[i][0] < item[j][0] and item[i][1] < item[j][1]):
                    pos_pair += 1
                elif (item[i][0] > item[j][0] and item[i][1] < item[j][1]) or \
                        (item[i][0] < item[j][0] and item[i][1] > item[j][1]):
                    neg_pair += 1
                else:
                    fair_pair += 1
    return {'pnr': pos_pair / neg_pair}


def evaluate_all_metric(qid_list, label_list, score_list, freq_list=None,file_dir=None,topN=None, save=None):
    cur_qid = qid_list[0]
    all_query = []
    tmp = []
    results_dict = {}
    for i in range(len(qid_list)):
        if qid_list[i]==-1:
            continue
        if qid_list[i] != cur_qid:
            if topN is not None:
                all_query.append(tmp[:topN])
            else:
                all_query.append(tmp)
            cur_qid = qid_list[i]
            tmp = []
        tmp.append([score_list[i], label_list[i]])
    if len(tmp)!=0:
        all_query.append(tmp[:topN])
    
    metrics_file=None
    if save:
        metrics_file={}
    
    dcg_all = calc_dcg(all_query, prefix='all',file_dir=None,metrics_file=metrics_file)
    ndcg_all = calc_ndcg(all_query, prefix='all',file_dir=None,metrics_file=metrics_file)
    err_all = calc_err(all_query, prefix='all',file_dir=None,metrics_file=metrics_file)
    pnr = calc_pnr(all_query)

    if save:
        import json
        with open(f'{file_dir}/metrics.json', 'w', encoding='utf-8') as f_out:
            json.dump(metrics_file, f_out, ensure_ascii=False, indent=2)
        
    
    if not freq_list:
        result_list = [dcg_all, ndcg_all,  pnr, err_all]
        for item in result_list:
            results_dict.update(item)
        if save:
            
            import pandas as pd
            df={}
            for k in results_dict:
                df[k]=[results_dict[k]]
            df=pd.DataFrame(df)
            df.to_csv(f'{file_dir}/metrics.csv',index=False,sep=',')
        return results_dict

    # evaluate on different frequency data
    cur_qid = qid_list[0]
    cur_freq = int(freq_list[0])
    high_freq_query = []
    mid_freq_query = []
    low_freq_query = []
    tmp = []
    for i in range(len(qid_list)):
        if qid_list[i] != cur_qid:
            if cur_freq == 0:
                high_freq_query.append(tmp)
            elif cur_freq == 1:
                mid_freq_query.append(tmp)
            elif cur_freq == 2:
                low_freq_query.append(tmp)
            # init
            cur_qid = qid_list[i]
            cur_freq = int(freq_list[i])
            tmp = []
        tmp.append([score_list[i], label_list[i]])

    if len(tmp) > 0:
        if cur_freq == 0:
            high_freq_query.append(tmp)
        elif cur_freq == 1:
            mid_freq_query.append(tmp)
        elif cur_freq == 2:
            low_freq_query.append(tmp)

    high_metrics,mid_metrics,low_metrics={},{},{}
    
    dcg_high_freq = calc_dcg(high_freq_query, prefix='high',file_dir=None,metrics_file=high_metrics)
    dcg_mid_freq = calc_dcg(mid_freq_query, prefix='mid',file_dir=None,metrics_file=mid_metrics)
    dcg_low_freq = calc_dcg(low_freq_query, prefix='low',file_dir=None,metrics_file=low_metrics)
    ndcg_high_freq = calc_ndcg(high_freq_query, prefix='high',file_dir=None,metrics_file=high_metrics)
    ndcg_mid_freq = calc_ndcg(mid_freq_query, prefix='mid',file_dir=None,metrics_file=mid_metrics)
    ndcg_low_freq = calc_ndcg(low_freq_query, prefix='low',file_dir=None,metrics_file=low_metrics)
    err_high_freq = calc_err(high_freq_query, prefix='high',file_dir=None,metrics_file=high_metrics)
    err_mid_freq = calc_err(mid_freq_query, prefix='mid',file_dir=None,metrics_file=mid_metrics)
    err_low_freq = calc_err(low_freq_query, prefix='low',file_dir=None,metrics_file=low_metrics)
    
    metrics_files=[high_metrics,mid_metrics,low_metrics]
    result_lists=[[dcg_high_freq, ndcg_high_freq, err_high_freq],[dcg_mid_freq, ndcg_mid_freq, err_mid_freq],[dcg_low_freq, ndcg_low_freq, err_low_freq]]
    freqs=['high','mid','low']
    if save:
        print('here')
        for i in range(3):
            results_dict={}
            result_list = result_lists[i]
            for item in result_list:
                results_dict.update(item)
            
            import json
            # print(f'{file_dir}/{freqs[i]}_metrics.json')
            with open(f'{file_dir}/{freqs[i]}_metrics.json', 'w', encoding='utf-8') as f_out:
                json.dump(metrics_files[i], f_out, ensure_ascii=False, indent=2)
            import pandas as pd
            df={}
            for k in results_dict:
                df[k]=[results_dict[k]]
            df=pd.DataFrame(df)
            df.to_csv(f'{file_dir}/{freqs[i]}_metrics.csv',index=False,sep=',')

    result_list = [dcg_all, dcg_high_freq, dcg_mid_freq, dcg_low_freq, ndcg_all, ndcg_high_freq,
                   ndcg_mid_freq, ndcg_low_freq, pnr, err_all, err_high_freq, err_mid_freq, err_low_freq]
    results_dict={}
    for item in result_list:
        results_dict.update(item)
    return results_dict
