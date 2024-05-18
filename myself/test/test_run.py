import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

from ir_measures import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

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



