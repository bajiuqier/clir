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
from typing import List, Union, cast
import ir_datasets
import ir_measures
from ir_measures import *
import pandas as pd
import numpy as np

from get_embedding import GetEmbedding
from modeling import EmbeddingModel
from argments import parse_args




# ----------------------------- searching -----------------------------
class searching():
    def __init__(self, tokenizer, model: EmbeddingModel, batch_size: int = 256) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.batch_size = batch_size

    def get_embedding(self, text: Union[List[str], str], tokenize_max_len: int = 512, convert_to_numpy: bool = True):
        self.model.eval()
        # input_was_string = False
        if isinstance(text, str):
            text = [text]
            # input_was_string = True

        all_embeddings = []
        for start_index in tqdm(range(0, len(text), self.batch_size), desc="Inference Embeddings", disable=len(text) < 256):
            text_batch = text[start_index:start_index + self.batch_size]

            inputs = self.tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=tokenize_max_len,
            ).to(self.device)

            embeddings = self.model(**inputs).embedding
            # 将 embeddings 转成 troch.Tensor 类型
            # embeddings = cast(torch.Tensor, embeddings)

            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.stack(all_embeddings)

        # 下面的操作返回的all_embeddings是一个一维的数据 但是在 do search 时 要求query是一个二维的向量 所以没必要做这一步
        # if input_was_string:
        #     return all_embeddings[0]
        return all_embeddings

    def index(
            self,
            documents: List[str],
            corpus_ids: List[int],
            index_factory: str = "Flat",
            # save_path: str = None,
            # save_embedding: bool = False,
            # load_embedding: bool = False,
    ): 
        assert type(documents) == list, 'corpus 的类型应该为 Union[List[str], str]'
        assert len(documents) == len(set(corpus_ids)), 'passage/document的数量和其对应的id的数量不相等 请进行检查'

        corpus_embeddings = self.get_embedding(text=documents, convert_to_numpy=True)
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
        # 然faiss 生成的 index 的 id 和我们的文档 id 相同
        faiss_index_map.add_with_ids(corpus_embeddings, np.array(corpus_ids))
        return faiss_index
    
    def search(
            self,
            queries: Union[List[str], str],
            faiss_index: faiss.Index,
            k: int = 20,
    ):
        
        query_embeddings = self.get_embedding(text=queries, convert_to_numpy=True)
        query_size = len(query_embeddings)
        
        all_scores = []
        all_indices = []
        for i in tqdm(range(0, query_size, self.batch_size), desc="Searching"):
            j = min(i + self.batch_size, query_size)
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
class evaluate():
    def __init__(self, queries: pd.DataFrame, qrels: pd.DataFrame, metrics: List) -> None:
        self.queries = queries
        self.qrels = qrels
        self.metrics = measures
        pass
    def generate_run(self, scores: np.array, indices: np.array) -> pd.DataFrame:
        '''
        scores: 
        indices: 
        run:    dataframe数据类型 表示检索结果 有也仅有 query_id doc_id score 3列数据 其中query_id doc_id为str类型 score为数值类型
        '''
        # 创建一个空列表来存储新的数据行
        rows = []
        
        for i, query_row in self.queries.iterrows():
            query_id = str(query_row['query_id'])
            for j in range(len(scores[i])):
                if indices[i][j] != -1: # 排除indice值为-1的情况
                    score = scores[i][j]
                    doc_id = str(indices[i][j])
                    rows.append([query_id, doc_id, score])
        
        run = pd.DataFrame(rows)
        run.columns = ['query_id', 'doc_id', 'score']

        return run


    def get_evaluates(self, run: pd.DataFrame):
        '''
        qrels:  dataframe类型的数据 应有也应仅有 query_id doc_id relevance 3列数据
                另外 query_id 和 doc_id 的值应该是 str 类型 relevance 的值应该是数值类型
                注意 注意 注意 应确保qrels中的query_id和run中的query_id的唯一值是对等的 不然会得到一个有误的结果
        run:    dataframe类型的数据 应有也应仅有 query_id doc_id score 3列数据
                另外 query_id 和 doc_id 的值应该是 str 类型 score 的值应该是数值类型
        metrics:    List 存放一个或者多个评价指标
        results:    结果是一个dict 但是返回的评测指标的顺序是随机的
        '''
        results = ir_measures.calc_aggregate(self.metrics, self.qrels, run)

        metric_results = {}
        for metric in METRICS_LIST:
            metric_results[str(metric)] = results[metric]

        return metric_results

# ----------------------------- evaluates by ir_measures -----------------------------


if __name__ == '__main__':
    args = parse_args() 
    METRICS_LIST = [R@5, R@10, nDCG@5, nDCG@10]
    path_home = Path.home().parent / 'mnt' / 'workspace'

    model_path = str(path_home / 'clir' / 'myself' / 'out_put' / 'embedding_model')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(model_path)

    dataset = ir_datasets.load('clirmatrix/kk/bi139-full/zh/test1')

    # 加载 目标语言 document 数据
    kk_document_path = str(path_home / 'Datasets' / 'mmarco' / 'english_collection_fragment.tsv')
    collections_df = pd.read_csv(kk_document_path, sep='\t', encoding='utf-8', names=['doc_id', 'text'])
    corpus_list = collections_df.loc[:]['text'].to_list()
    corpus_ids_list = collections_df.loc[:]['doc_id'].to_list()
    # 将 passage/document 对应的 id 转成 int 类型
    corpus_ids_list = list(map(int, corpus_ids_list))

    # 加载 源语言 queries 数据
    queries_df = pd.DataFrame(dataset.queries_iter())
    queries_list = queries_df.loc[:]['text'].to_list()
    queries_df['query_id'] = queries_df['query_id'].astype(str)
    queries_ids_list = queries_df.loc[:]['query_id'].to_list()
    queries_ids_list = list(map(int, queries_ids_list))

    # 加载 qrels 数据
    qrels_df = pd.DataFrame(dataset.qrels_iter())
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

    print(results)



