from pathlib import Path
import jsonlines
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import random

HOME_DIR = Path(__file__).parent / 'data_file'

train_dataset_file_path = str(HOME_DIR / 'train_dataset.jsonl')
triples_file_path = str(HOME_DIR / 'base_train_triplet_id_final.csv')
adj_item_info_file_path = str(HOME_DIR / 'base_train_adj_item_info_filled.csv')

train_dataset_df = pd.read_json(train_dataset_file_path, lines=True, encoding='utf-8')
train_dataset_df['query_id'] = train_dataset_df['query_id'].astype(str)

triples_df = pd.read_csv(triples_file_path, encoding='utf-8')
adj_item_info_df = pd.read_csv(adj_item_info_file_path, encoding='utf-8')


# 定义一个函数来提取 adj_item_info 中指定的前 n 个元素
def extract_adj_item_info(row, n):
    adj_item_info = row["adj_item_info"]
    new_adj_item_info = {k: v[:n] for k, v in adj_item_info.items()}
    return new_adj_item_info


n = 8
adj_item_num = 3
print(f"生成train_dataset_with_{adj_item_num + 1}_adj_item_df")
for index, row in tqdm(train_dataset_df.iterrows(), total=train_dataset_df.shape[0]):
    existing_adj_item_info = row['adj_item_info']
    q_item_qid = row['q_item_qid']

    # 从 triples 中获取所有相邻的 adj_item_qid
    adj_item_qids = triples_df[triples_df['q_item_qid'] == q_item_qid]['adj_item_qid'].tolist()

    # 过滤掉已存在的 adj_item_qid，避免重复
    # label_zh的数据来依次增加相邻实体相关信息的数量
    existing_labels_zh = set(existing_adj_item_info.get('label_zh'))

    # 收集新的信息
    new_adj_info = {key: value[:] for key, value in existing_adj_item_info.items()}

    count = len(new_adj_info.get('label_zh'))

    adj_item_info_adding_df = adj_item_info_df[adj_item_info_df['qid'].isin(random.sample(adj_item_qids, k=n + 3))]
    ddd = set(adj_item_info_adding_df['label_zh'].to_list()) - existing_labels_zh
    adj_item_info_adding: pd.Series = adj_item_info_adding_df[adj_item_info_adding_df['label_zh'].isin(ddd)].sample(1).iloc[0]


    row['label_zh'].append(adj_item_info_adding_df['label_zh'])
    row['label_kk'].append(adj_item_info_adding_df['label_kk'])
    row['description_zh'].append(adj_item_info_adding_df['description_zh'])
    row['description_kk'].append(adj_item_info_adding_df['description_kk'])

    adj_item_num += 1

file_path = str(HOME_DIR / f"train_dataset_with_{adj_item_num}_adj_item.jsonl")
train_dataset_df.to_json(file_path, orient="records", lines=True, force_ascii=False)


# for adj_item_num in tqdm(range(1, 8)):
#     if adj_item_num <= 3:
#         train_dataset_with_n_adj_item_df = train_dataset_df.copy(deep=True)
#         train_dataset_with_n_adj_item_df["adj_item_info"] = train_dataset_with_n_adj_item_df.apply(
#             lambda row: extract_adj_item_info(row, adj_item_num), axis=1)
#
#         # 将每个新的 DataFrame 保存为 JSONL 文件
#         # 指定 force_ascii=False 保留 Unicode 字符
#         file_path = str(HOME_DIR / f"train_dataset_with_{adj_item_num}_adj_item.jsonl")
#         train_dataset_with_n_adj_item_df.to_json(file_path, orient="records", lines=True, force_ascii=False)
#     else:
#         pass

