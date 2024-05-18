import ir_measures
from ir_measures import *
import pandas as pd
import numpy as np
from typing import List


# ----------------------------- evaluates by ir_measures -----------------------------
def generate_run(queries: pd.DataFrame, scores: np.array, indices: np.array) -> pd.DataFrame:
    '''
    queries: 
    scores: 
    indices: 
    run: dataframe数据类型 表示检索结果 有也仅有 query_id doc_id score 3列数据 其中query_id doc_id为str类型 score为数值类型
    '''
    # 创建一个空列表来存储新的数据行
    rows = []
    
    for i, query_row in queries.iterrows():
        query_id = str(query_row['query_id'])
        for j in range(len(scores[i])):
            if indices[i][j] != -1: # 排除indice值为-1的情况
                score = scores[i][j]
                doc_id = str(indices[i][j])
                rows.append([query_id, doc_id, score])
    
    run = pd.DataFrame(rows)
    run.columns = ['query_id', 'doc_id', 'score']

    return run


def get_evaluates(qrels: pd.DataFrame, run: pd.DataFrame, metrics: List):
    '''
    qrels: dataframe类型的数据 应有也应仅有 query_id doc_id relevance 3列数据
    另外 query_id 和 doc_id 的值应该是 str 类型 relevance 的值应该是数值类型
    注意 注意 注意 应确保qrels中的query_id和run中的query_id的唯一值是对等的 不然会得到一个有误的结果
    run: dataframe类型的数据 应有也应仅有 query_id doc_id score 3列数据
    另外 query_id 和 doc_id 的值应该是 str 类型 score 的值应该是数值类型
    metrics: List 存放一个或者多个评价指标
    results: 结果是一个dict 但是返回的评测指标的顺序是随机的
    '''
    results = ir_measures.calc_aggregate(metrics, qrels, run)
    return results

# ----------------------------- evaluates by ir_measures -----------------------------


# ----------------------------- evaluates by custom functions -----------------------------
def evaluate_function(preds, labels, cutoffs=[1,10,100]):
    """
    Evaluate MRR and Recall at cutoffs.
    """
    metrics = {}
    
    # MRR
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[k] += 1 / i
                jump = True
            if jump:
                break
    mrrs /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        mrr = mrrs[i]
        metrics[f"MRR@{cutoff}"] = mrr

    # Recall
    recalls = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        for k, cutoff in enumerate(cutoffs):
            recall = np.intersect1d(label, pred[:cutoff])
            recalls[k] += len(recall) / len(label)
    recalls /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        recall = recalls[i]
        metrics[f"Recall@{cutoff}"] = recall

    return metrics

# ----------------------------- evaluates by custom functions -----------------------------

