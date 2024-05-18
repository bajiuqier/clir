from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json

triples_train_ids_path = Path.home() / 'Desktop' / 'Datasets' / 'mmarco' / 'v2' / 'triples.train.ids.small.tsv'
chinese_collection_path = Path.home() / 'Desktop' / 'Datasets' / 'mmarco' / 'v2' / 'google' / 'collenctions' / 'chinese_collection.tsv'
english_collection_path = Path.home() / 'Desktop' / 'Datasets' / 'mmarco' / 'v2' / 'google' / 'collenctions' / 'english_collection.tsv'
chinese_queries_path = Path.home() / 'Desktop' / 'Datasets' / 'mmarco' / 'v2' / 'google' / 'queries' / 'chinese_queries.train.tsv'
english_queries_path = Path.home() / 'Desktop' / 'Datasets' / 'mmarco' / 'v2' / 'google' / 'queries' / 'english_queries.train.tsv'
run_bm25_chinese_msmarco_path = Path.home() / 'Desktop' / 'Datasets' / 'mmarco' / 'v2' / 'google' / 'runs' / 'run.bm25_chinese-msmarco.txt'


# 存储数据文件
data_length = 100000
zh_en_triple_data_v2_file = Path.home() / 'Desktop' / 'Datasets' / 'mmarco' / f'zh_en_triple_data_v2_{data_length}.jsonl'

collection_df_col_names = ['passage_id', 'text']
queries_df_col_names = ['query_id', 'text']
# 从hunggingface中下载的triples数据中已经包含了query对应的pos和neg样本id
triples_train_ids_df_col_names = ['query_id', 'positive_id', 'negative_id']
chinese_collection_df_col_names = collection_df_col_names
english_collection_df_col_names = collection_df_col_names
chinese_queries_df_col_names = queries_df_col_names
english_queries_df_col_names = queries_df_col_names

# 加载三元组id数据
triples_train_ids_df = pd.read_csv(triples_train_ids_path, sep='\t', encoding='utf-8', names=triples_train_ids_df_col_names)
# 加载中文query df数据
chinese_queries_df = pd.read_csv(chinese_queries_path, sep='\t', encoding='utf-8', names=chinese_queries_df_col_names)
# 加载英文passage df数据
english_collection_df = pd.read_csv(english_collection_path, sep='\t', encoding='utf-8', names=english_collection_df_col_names)

# 将 query_id 和 passage_id 设置为 index，以便可以快速的访问其对应的 text
chinese_queries_df.set_index('query_id', inplace=True)
# 判断df中时候存在重复的index 存在返回True 不存在返回False
assert chinese_queries_df.index.duplicated().any() == False, 'query dataframe数据中 存在重复的index 请检查'
english_collection_df.set_index('passage_id', inplace=True)
assert english_collection_df.index.duplicated().any() == False, 'collection dataframe数据中 存在重复的index 请检查'

# 创建triple数据的基本样式 然后依次修改里面的值 再进行存储
triples_data = {'query': '', 'positive': '', 'negative': ''}

with open(zh_en_triple_data_v2_file, 'w', encoding='utf-8') as f:
    for index, row in tqdm(triples_train_ids_df.iterrows(), total=triples_train_ids_df.shape[0]):
        query_id = int(row['query_id'])
        positive_id = int(row['positive_id'])
        negative_id = int(row['negative_id'])

        triples_data['query'] = chinese_queries_df.loc[query_id, 'text']
        triples_data['positive'] = english_collection_df.loc[positive_id, 'text']
        triples_data['negative'] = english_collection_df.loc[negative_id, 'text']

        json.dump(triples_data, f, ensure_ascii=False)
        f.write('\n')


        if index == data_length - 1:
            break