import faiss
import numpy as np
from tqdm import tqdm
from typing import List, Union

from get_embedding import GetEmbedding


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