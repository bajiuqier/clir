import ir_datasets
import pandas as pd
from pathlib import Path

HOME_DIR = Path(__file__).parent / 'data_file'

qid_file = str(HOME_DIR.parent / 'query_entity_info' / 'full_train_QID_filtered_search_results.csv')
qid_df = pd.read_csv(qid_file, encoding='utf-8')

# 
CLIRMatrix_dataset = ir_datasets.load('clirmatrix/kk/bi139-full/zh/train') 
# docstore = CLIRMatrix_dataset.docs_store()
queries_df = pd.DataFrame(CLIRMatrix_dataset.queries_iter())


# 将 queries_df 和 qid_df 合并 然后删除 id 为空的行
merge_df = queries_df.merge(qid_df, left_on='text', right_on='query', how='left')
merge_df.dropna(subset=['id'], how='any', inplace=True)
merge_df = merge_df[['query_id', 'text', 'id']]
merge_df.drop_duplicates(inplace=True)
merge_df.rename(columns={'id': 'qid'}, inplace=True)

# 再将处理好的 merige_df 和 item_info_df 合并 然后删除 中文和哈萨克语信息 为空的数据
item_info_file = str(HOME_DIR / 'item_info.csv')
item_info_df = pd.read_csv(item_info_file, encoding='utf-8')
merge_df = pd.merge(merge_df, item_info_df, left_on='qid', right_on='item', how='left')
merge_df.dropna(subset=['label_zh', 'label_kk', 'description_zh', 'description_kk'], how='any', inplace=True)
# 存储 query2qid 文件
query2qid_file = str(HOME_DIR / 'query2qid.csv')
query2qid_df = merge_df[['query_id', 'text', 'qid']].copy()
query2qid_df.drop_duplicates(inplace=True)
query2qid_df.to_csv(query2qid_file, index=False, encoding='utf-8')


# 构建新的 qrels
qrels_file = str(HOME_DIR / 'qrels.csv')
qrels_df = pd.DataFrame(CLIRMatrix_dataset.qrels_iter())
# 删除点 qrels 中多余的 query
query_ids = set(query2qid_df['query_id'])
qrels_filtered = qrels_df[qrels_df['query_id'].isin(query_ids)]

# 统计每个 query_id 对应的 doc_id 数量
query_counts = qrels_filtered['query_id'].value_counts()

# 准备一个新 DataFrame 来存放结果
result_df = pd.DataFrame(columns=qrels_filtered.columns)

# 遍历每个 query_id 并进行处理
for query_id, count in query_counts.items():
    query_df = qrels_filtered[qrels_filtered['query_id'] == query_id]
    
    if count < 100:
        # 从其他 query_id 对应的 doc_id 中随机选择补充
        remaining = 100 - count
        other_docs = qrels_filtered[qrels_filtered['query_id'] != query_id]['doc_id'].sample(remaining, replace=True)
        new_rows = pd.DataFrame({
            'query_id': [query_id] * remaining,
            'doc_id': other_docs.values,
            'relevance': [0] * remaining,
            'iteration': [0] * remaining
        })
        query_df = pd.concat([query_df, new_rows])
        
    elif count > 100:
        # 删除多余的数据，并更改最后 5 个 doc_id
        query_df = query_df.head(95)
        other_docs = qrels_filtered[qrels_filtered['query_id'] != query_id]['doc_id'].sample(5, replace=True)
        new_rows = pd.DataFrame({
            'query_id': [query_id] * 5,
            'doc_id': other_docs.values,
            'relevance': [0] * 5,
            'iteration': [0] * 5
        })
        query_df = pd.concat([query_df, new_rows])
        
    # 添加到结果 DataFrame
    result_df = pd.concat([result_df, query_df])

result_df.to_csv(qrels_file, index=False, encoding='utf-8')