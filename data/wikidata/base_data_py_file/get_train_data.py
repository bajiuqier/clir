import pandas as pd
from pathlib import Path
import ir_datasets
import random
import jsonlines
from typing import NamedTuple
from tqdm import tqdm


def filter_qrels(qrels_df: pd.DataFrame, filter_reference_file: str) -> pd.DataFrame:

    # 加载 过滤 所需的文件
    filter_reference_df = pd.read_csv(filter_reference_file, encoding='utf-8').astype(str)
    query_ids = set(filter_reference_df['query_id'])

    # 过滤掉多余的 query_id 的文档数据
    qrels_filtered_df = qrels_df[qrels_df['query_id'].isin(query_ids)]
    # 删除 iteration 列数据
    qrels_filtered_df.drop('iteration', axis=1, inplace=True)

    return qrels_filtered_df

def build_new_base_train_qrels(original_qrels: pd.DataFrame, new_qrels_file: str=None, save_new_qrels: bool=True) -> pd.DataFrame:
    '''
    base 版本数据的 qrels 中，作者已经为每个 query 构建了 100 个对应的文档
    其中 相关度从高到底 6 5 4 3 2 1 0
    full 版本数据的 qrels 和 base不同的是 query 没有对应的负样本文档 及 没有相关度为 0 的文档 id
    '''
    # 设置随机种子
    random_seed = 42
    random.seed(random_seed)

    # 计算每个 query_id 对应的 relevance 为 0 的 doc_id 数量
    neg_samples_counts_df = original_qrels[original_qrels['relevance'] == 0].groupby('query_id').size().reset_index(name='neg_sample_count')
    # query_id 对应的 relevance 为 0 的 doc_id 数量 不足 5个的 query_id
    insufficient_neg_samples_df = neg_samples_counts_df[neg_samples_counts_df['neg_sample_count'] < 5]
    # 获取 这些 query_ids
    query_ids = insufficient_neg_samples_df['query_id'].unique()
    # 准备一个包含所有doc_id的列表，用于随机选择
    all_doc_ids = original_qrels['doc_id'].unique()

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
    original_qrels['query_id'] = original_qrels['query_id'].astype(int)
    original_qrels['relevance'] = original_qrels['relevance'].astype(int)
    new_qrels = original_qrels.sort_values(by=['query_id', 'relevance'], ascending=[True, False]).reset_index(drop=True)

    new_qrels['query_id'] = new_qrels['query_id'].astype(str)
    new_qrels['doc_id'] = new_qrels['doc_id'].astype(str)
    new_qrels['relevance'] = new_qrels['relevance'].astype(int)

    if save_new_qrels:
        new_qrels.to_csv(new_qrels_file, index=False, encoding='utf-8')
        print("------------------------------------------------------")
        print(f"新的 qrels 文件已经保存在{new_qrels_file}")
        print("------------------------------------------------------")
        
    return new_qrels

def build_train_data(docstore: NamedTuple, query_qid_file: str, qrels_file: str, item_info_file: str, triple_id_file: str):
    '''
    构建 训练 使用的 jsonl 数据    
    '''
    query_qid_df = pd.read_csv(query_qid_file, encoding='utf-8')

    qrels_df = pd.read_csv(qrels_file, encoding='utf-8')

    item_info_df = pd.read_csv(item_info_file, encoding='utf-8')

    triple_id_df = pd.read_csv(triple_id_file, encoding='utf-8')

    # 使用 ir_datasets 加载文档内容
    CLIRMatrix_dataset = ir_datasets.load('clirmatrix/kk/bi139-full/zh/train') 
    docstore = CLIRMatrix_dataset.docs_store()

    # 构建JSONL数据
    jsonl_data = []

    adjitem_num = 3
    size = int(0.8 * len(query2qid))
    train_query2qid = query2qid.iloc[:size]
    test_query2qid = query2qid.iloc[size:]

    for _, query_row in test_query2qid.iterrows():
        query_id = query_row['query_id']
        query_text = query_row['text']
        query_qid = query_row['qid']
        
        # 获取查询对应实体的信息
        q_item = item_info[item_info['item'] == query_qid].iloc[0]

        # 检查q_item中的'label_zh', 'label_kk' description_zh description_kk 是否有一个为空
        if q_item[['label_zh', 'label_kk', 'description_zh', 'description_kk']].isnull().any():
            continue

        q_item_info = {
            "label_zh": q_item['label_zh'],
            "label_kk": q_item['label_kk'],
            "description_zh": q_item['description_zh'],
            "description_kk": q_item['description_kk']
        }
        
        # 获取相邻实体的信息
        # adj_items = filtered_triplet_id[filtered_triplet_id['item'] == qid]['adjItem'].unique()
        adj_items = triplet_id[triplet_id['item'] == query_qid]['adjItem']

        # 舍弃相邻实体数量为0的query
        if len(adj_items) == 0:
            continue
        elif len(adj_items) > 0 and len(adj_items) < adjitem_num:
            adj_items = adj_items.tolist()
            while len(adj_items) < adjitem_num:
                adj_items.append(adj_items[(adjitem_num - len(adj_items)) % len(adj_items)])
            adj_items = pd.Series(adj_items)
        else:
            # replace=False 是 pandas sample() 方法的一个参数，表示在抽样时不进行重复抽样
            # sampled_adj_items = adj_items.sample(adjitem_num, replace=False)
            adj_items = adj_items.sample(adjitem_num)

        adj_item_info = {
            "label_zh": [],
            "label_kk": [],
            "description_zh": [],
            "description_kk": []
        }
        
        for adj_item in adj_items:
            adj_info = adjitem_info[adjitem_info['item'] == adj_item].iloc[0]
            # 这里可以判断一下 adj_item 中的下面的信息是否 为空
            adj_item_info["label_zh"].append(adj_info['label_zh'])
            adj_item_info["label_kk"].append(adj_info['label_kk'])
            adj_item_info["description_zh"].append(adj_info['description_zh'])
            adj_item_info["description_kk"].append(adj_info['description_kk'])
        

        query_docs = qrels[qrels['query_id'] == query_id]
        if len(query_docs) == 0:
            continue
        else:
            pos_doc_ids = query_docs[query_docs['relevance'] != 0]['doc_id'][:3]
            # neg_doc_ids = query_docs[query_docs['relevance'] == 0]['doc_id'][:3]
            neg_doc_ids = query_docs[query_docs['relevance'] == 0]['doc_id'].sample(3)

            
            pos_doc_texts = [docstore.get(doc_id).text for doc_id in pos_doc_ids]
            neg_doc_texts = [docstore.get(doc_id).text for doc_id in neg_doc_ids]
        
        # if len(pos_doc_texts) == 0:
        #     continue
        
        jsonl_data.append({
            "query_id": query_id,
            "query": query_text,
            "pos_doc": pos_doc_texts,
            "neg_doc": neg_doc_texts,
            "q_item_info": q_item_info,
            "adj_item_info": adj_item_info
        })

    # 将数据写入JSONL文件
    with jsonlines.open(test_dataset_file, mode='w') as writer:
        writer.write_all(jsonl_data)



    pass




if __name__ == "__main__":

    HOME_DIR = Path(__file__).parent.parent / 'base_data'

    # 加载 zh-kk clir 数据集
    CLIRMatrix_dataset = ir_datasets.load('clirmatrix/kk/bi139-base/zh/train')
    # 加载原始查询数据
    # queries_df = pd.DataFrame(CLIRMatrix_dataset.queries_iter())
    # 加载 doc 数据
    docstore = CLIRMatrix_dataset.docs_store()
    # 加载原始的 qrels 数据
    qrels_df = pd.DataFrame(CLIRMatrix_dataset.qrels_iter())

    # 加载过滤 qrels 数据所需的参考文件
    triple_id_file = str(HOME_DIR / 'base_train_triplet_id_fragment_5.csv')

    qrels_filtered_df = filter_qrels(qrels_df=qrels_df, filter_reference_file=triple_id_file)
    # 过滤后 query 数量为 5092
    # 段落数量 201707

    new_qrels_file = str(HOME_DIR / 'base_train_qrels.csv')
    new_qrels_df = build_new_base_train_qrels(qrels_filtered_df, new_qrels_file, save_new_qrels=False)
    print(new_qrels_df)

