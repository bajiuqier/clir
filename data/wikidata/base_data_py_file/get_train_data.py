import pandas as pd
from pathlib import Path
import ir_datasets
import random
from tqdm import tqdm


HOME_DIR = Path(__file__).parent.parent / 'base_data'

triple_id_file = str(HOME_DIR / 'base_train_triplet_id_fragment_5.csv')
new_qrels_file = str(HOME_DIR / 'base_train_qrels.csv')

# 加载 zh-kk clir 数据集
CLIRMatrix_dataset = ir_datasets.load('clirmatrix/kk/bi139-base/zh/train')

# queries_df = pd.DataFrame(CLIRMatrix_dataset.queries_iter())
# docstore = CLIRMatrix_dataset.docs_store()

# 加载原始的 qrels 数据
qrels_df = pd.DataFrame(CLIRMatrix_dataset.qrels_iter())
# 加载经过 获取过三元组数据的 query_id 数据
triple_id_df = pd.read_csv(triple_id_file, encoding='utf-8').astype(str)
# 取出 处理过后的数据中的 query_id 的唯一值
query_ids = set(triple_id_df['query_id'])

# 过滤掉多余的 query_id 的文档数据
qrels_filtered_df = qrels_df[qrels_df['query_id'].isin(query_ids)]


def build_new_base_train_qrels(original_qrels: pd.DataFrame, new_qrels_file: str=None, save_new_qrels: bool=True) -> pd.DataFrame:
    '''
    base 版本数据的 qrels 中，作者已经为每个 query 构建了 100 个对应的文档
    其中 相关度从高到底 6 5 4 3 2 1 0
    full 版本数据的 qrels 和 base不同的是 query 没有对应的负样本文档 及 没有相关度为 0 的文档 id
    '''

    # 计算每个query_id对应的relevance为0的doc_id数量
    # zero_counts = original_qrels[original_qrels['relevance'] == 0].groupby('query_id').size().reset_index(name='zero_count')
    
    # 设置随机种子
    random_seed = 42
    random.seed(random_seed)
    # 获取 query_ids
    query_ids = original_qrels['query_id'].unique()
    # 准备一个包含所有doc_id的列表，用于随机选择
    all_doc_ids = qrels_df['doc_id'].unique()

    # 定义一个列表 存储 添加的负样本
    new_rows = []

    for query_id in tqdm(query_ids, total=len(query_ids)):
        # 获取当前 query_id 对应的行
        current_docs = original_qrels[original_qrels['query_id'] == query_id]
        # 获取当前 query_id 下相关度为 0 的 doc_id
        zero_relevance_docs = current_docs[current_docs['relevance'] == 0]['doc_id']

        # 如果数量少于5，补足缺少的 doc_id
        if len(zero_relevance_docs) < 5:
            # 当前缺少的 doc_id 数量
            shortage = 5 - len(zero_relevance_docs)
            
            # 获取当前 query_id 下所有的 doc_id
            current_doc_ids = current_docs['doc_id'].unique()

            # 从其他查询中随机选择不足的数量的文档ID
            # 并确保随机选择的文档ID不会与当前查询下的文档ID重复
            candidates_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id not in current_doc_ids]
            
            # 随机选择 shortage 数量的 doc_id
            selected_docs = random.sample(list(candidates_doc_ids), shortage)
            
            # 记录新添加的数据
            new_rows.extend([{'query_id': query_id, 'doc_id': doc, 'relevance': 0} for doc in selected_docs])

            # 如果添加了相关度为0的文档，则删除相关度最小的文档ID
            # 获取需要删除的行
            docs_to_remove = current_docs.iloc[:100 - len(zero_relevance_docs)].tail(shortage)
            # 删除这些行
            original_qrels = original_qrels.drop(docs_to_remove.index)

    if len(new_rows) > 0:
        original_qrels = pd.concat([original_qrels, pd.DataFrame(new_rows)], ignore_index=True)

    # 按 query_id 和 relevance 排序，保证每个 query_id 对应的100个 doc_id 是连续的
    new_qrels = original_qrels.sort_values(by=['query_id', 'relevance'], ascending=[True, False]).reset_index(drop=True)

    if save_new_qrels:
        new_qrels.to_csv(new_qrels_file, index=False, encoding='utf-8')
        
    return new_qrels

new_qrels_df = build_new_base_train_qrels(qrels_filtered_df, new_qrels_file, save_new_qrels=True)




