import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import faiss
import numpy as np
from tqdm import tqdm
from typing import List, Union
import ir_measures
from ir_measures import *
import pandas as pd
import numpy as np


from get_embedding import GetEmbedding
from searching import index, search
from evaluate import generate_run, get_evaluates


# ------------------------ arguments ------------------------
path_home = Path.home().parent / 'mnt' / 'workspace'

checkpoint = str(path_home / 'models' / 'out_put' / 'savechpt-30.0')
xlmr_model_path = str(path_home / 'models' / 'xlm-roberta-base')

english_collection_path = str(path_home / 'Datasets' / 'mmarco' / 'english_collection_fragment.tsv')
english_collection_embedding_path = str(path_home / 'Datasets' / 'mmarco' / 'english_collection_embedding.npy')
# chinese_queries_dev_fragment_path = str(path_home / 'Datasets' / 'mmarco' / 'chinese_queries_dev_fragment.tsv')
chinese_queries_dev_small_path = str(path_home / 'Datasets' / 'mmarco' / 'chinese_queries.dev.small.tsv')
qrels_dev_small_path = str(path_home / 'Datasets' / 'mmarco' / 'qrels.dev.small.tsv')

query_max_length = 32
passage_max_length = 128

METRICS_LIST = [R@5, R@10, nDCG@5, nDCG@10]
# ------------------------ arguments ------------------------

# ----------------------------- searching -----------------------------
def index(encoder: GetEmbedding, corpus: Union[List[str], str], corpus_ids: List[int], index_factory: str = "Flat", load_embedding: bool = False, embedding_path: str = None):
    '''
    encoder:
    corpus:
    corpus_ids:
    index_factory:
    load_embedding:
    embedding_path:
    '''
    if isinstance(corpus, str):
        assert len([corpus]) == len(set(corpus_ids)), 'passage/document的数量和其对应的id的数量不相等 请进行检查'
    else:
        assert len(corpus) == len(set(corpus_ids)), 'passage/document的数量和其对应的id的数量不相等 请进行检查'

    if load_embedding:
        test = encoder.encode("test")
        dtype = test.dtype
        dim = len(test)

        corpus_embeddings = np.memmap(
            embedding_path,
            mode="r",
            dtype=dtype
        ).reshape(-1, dim)
    else:
        corpus_embeddings = encoder.encode(sentences=corpus, convert_to_numpy=True)
        dim = corpus_embeddings.shape[-1]
    
    # create faiss index
    faiss_index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)
    faiss_index_map = faiss.IndexIDMap(faiss_index)

    # if model.device == torch.device("cuda"):
    #     # co = faiss.GpuClonerOptions()
    #     co = faiss.GpuMultipleClonerOptions()
    #     co.useFloat16 = True
    #     # faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index, co)
    #     faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    # NOTE: faiss only accepts float32
    print("Adding embeddings...")
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index_map.train(corpus_embeddings)
    faiss_index_map.add_with_ids(corpus_embeddings, np.array(corpus_ids))
    return faiss_index
 
def search(encoder: GetEmbedding, queries: Union[List[str], str], faiss_index: faiss.Index, k: int = 20, query_batch_size: int=256):
    
    query_embeddings = encoder.encode(sentences=queries, convert_to_numpy=True)
    query_size = len(query_embeddings)
    
    all_scores = []
    all_indices = []
    for i in tqdm(range(0, query_size, query_batch_size), desc="Searching"):
        j = min(i + query_batch_size, query_size)
        query_embedding = query_embeddings[i: j]
        # indice 存放检索出向量的索引
        # score 存放查询与检索出向量的相似度得分
        # faiss 检索出来的是一个排序好的结果，相似度越高 排名越靠前
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=k)
        all_scores.append(score)
        all_indices.append(indice)
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices

# ----------------------------- searching -----------------------------


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


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained(xlmr_model_path)
    model = AutoModel.from_pretrained(checkpoint)

    get_embedding = GetEmbedding(
        model=model,
        tokenizer=tokenizer,
        normalize_embeddings=True,
        pooling_method='cls',
        batch_size=256,
        max_seq_length=256
        )

    # 加载 目标语言 passage/document 数据
    collections_df = pd.read_csv(english_collection_path, sep=',', encoding='utf-8', names=['passage_id', 'text'])
    corpus_list = collections_df.loc[:]['text'].to_list()
    corpus_ids_list = collections_df.loc[:]['passage_id'].to_list()
    # 将 passage/document 对应的 id 转成 int 类型
    corpus_ids_list = list(map(int, corpus_ids_list))

    # 加载 源语言 queries 数据
    queries_df = pd.read_csv(chinese_queries_dev_small_path, sep='\t', encoding='utf-8', names=['query_id', 'text'])
    queries_list = queries_df.loc[:]['text'].to_list()
    queries_df['query_id'] = queries_df['query_id'].astype(str)
    queries_ids_list = queries_df.loc[:]['query_id'].to_list()
    queries_ids_list = list(map(int, queries_ids_list))

    # 加载 qrels 数据
    qrels_df = pd.read_csv(qrels_dev_small_path, sep='\t', encoding='utf-8', names=['query_id', 'iteration', 'doc_id', 'relevance'])
    qrels_df = qrels_df.drop(columns=['iteration'])
    qrels_df['query_id'] = qrels_df['query_id'].astype(str)
    qrels_df['doc_id'] = qrels_df['doc_id'].astype(str)
    qrels_query_ids_list = qrels_df.loc[:]['query_id'].to_list()
    qrels_query_ids_list = list(map(int, qrels_query_ids_list))

    assert set(queries_ids_list) == set(qrels_query_ids_list), 'qrels中的query和queries中的query 数量/内容存在差异 请检查 避免评测结果出现误差'


    # 构建索引
    faiss_index = index(
        encoder=get_embedding,
        corpus=corpus_list,
        corpus_ids=corpus_ids_list,
        load_embedding=False,
        embedding_path=english_collection_embedding_path
        )
    
    assert faiss_index.ntotal == len(corpus_ids_list), '构建的索引和passage/document库中的数据量不对等 请检查'
    
    # 检索
    scores, indices = search(
        encoder=get_embedding,
        queries=queries_list,
        faiss_index=faiss_index,
        k=20,
        query_batch_size=256
    )

    # 构建通过 ir-measures 评估所需的 run 数据
    run_df = generate_run(
        queries=queries_df,
        scores=scores,
        indices=indices
    )

    # 评估
    results = get_evaluates(
        qrels=qrels_df,
        run=run_df,
        metrics=METRICS_LIST
    )

    metric_results = {}
    for metric in METRICS_LIST:
        metric_results[str(metric)] = results[metric]

    print(metric_results)



