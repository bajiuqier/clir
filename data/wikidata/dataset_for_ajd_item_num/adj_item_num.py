from pathlib import Path
import jsonlines
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import random

HOME_DIR = Path(__file__).parent / 'data_file'

train_dataset_file_path = str(HOME_DIR / 'train_dataset.jsonl')
test_dataset_file_path = str(HOME_DIR / 'test_dataset.jsonl')
triples_file_path = str(HOME_DIR / 'base_train_triplet_id_final.csv')
adj_item_info_file_path = str(HOME_DIR / 'base_train_adj_item_info_filled.csv')

train_dataset_df = pd.read_json(train_dataset_file_path, lines=True, encoding='utf-8')
train_dataset_df['query_id'] = train_dataset_df['query_id'].astype(str)

# test_dataset_df = pd.read_json(test_dataset_file_path, lines=True, encoding='utf-8')
# test_dataset_df['query_id'] = test_dataset_df['query_id'].astype(str)

triples_df = pd.read_csv(triples_file_path, encoding='utf-8')
adj_item_info_df = pd.read_csv(adj_item_info_file_path, encoding='utf-8')


# 定义一个函数来提取 adj_item_info 中指定的前 n 个元素
def extract_adj_item_info(row, n):
    adj_item_info = row["adj_item_info"]
    new_adj_item_info = {k: v[:n] for k, v in adj_item_info.items()}
    return new_adj_item_info


adj_item_num = 3
print(f"生成train_dataset_with_{adj_item_num+1}_adj_item_df")

for index, row in tqdm(train_dataset_df.iterrows(), total=train_dataset_df.shape[0]):
    existing_adj_item_info = row['adj_item_info']
    q_item_qid = row['q_item_qid']

    # 从 triples 中获取所有相邻的 adj_item_qid
    adj_item_qids = triples_df[triples_df['q_item_qid'] == q_item_qid]['adj_item_qid'].tolist()

    # 获取当前 q_item_qid 所有相邻实体的信息
    # adj_item_info_adding_df = adj_item_info_df[adj_item_info_df['qid'].isin(random.sample(adj_item_qids, k=n + 3))]
    adj_item_info_df = adj_item_info_df[adj_item_info_df['qid'].isin(adj_item_qids)]

    # 计算训练数据中 当前 q_item_qid 已经存在的相邻实体的数量
    the_num_existing_adj_item_info = len(existing_adj_item_info.get('label_zh'))
    # 计算当前 q_item_qid 所有相邻实体的数量
    the_num_all_adj_item_info = adj_item_info_df.shape[0]

    # 判断相邻实体的数量是否足够 来为 q_item_qid 添加新的相邻实体信息
    if the_num_existing_adj_item_info >= the_num_all_adj_item_info:
        # 如果相邻实体数量不足，则从 adj_item_info_df 中随机选择 1 个相邻实体
        adj_item_info_adding: pd.Series = adj_item_info_df.sample(1).iloc[0]
    else:
        # 根据 label_zh 的数据来依次增加相邻实体相关信息的数量
        existing_labels_zh = set(existing_adj_item_info.get('label_zh'))
        ddd = list(set(adj_item_info_df['label_zh'].to_list()) - existing_labels_zh)

        adj_item_info_adding: pd.Series = \
        adj_item_info_df[adj_item_info_df['label_zh'].isin(ddd)].sample(1).iloc[0]

    row['label_zh'].append(adj_item_info_adding['label_zh'])
    row['label_kk'].append(adj_item_info_adding['label_kk'])
    row['description_zh'].append(adj_item_info_adding['description_zh'])
    row['description_kk'].append(adj_item_info_adding['description_kk'])



file_path = str(HOME_DIR / f"train_dataset_with_{adj_item_num+1}_adj_item.jsonl")
train_dataset_df.to_json(file_path, orient="records", lines=True, force_ascii=False)


# 构建相邻实体数量为 1 2 3 的训练数据
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


# 构建相邻实体数量为 1 2 3 的测试数据
# for adj_item_num in tqdm(range(1, 8)):
#     if adj_item_num <= 3:
#         test_dataset_with_n_adj_item_df = test_dataset_df.copy(deep=True)
#         test_dataset_with_n_adj_item_df["adj_item_info"] = test_dataset_with_n_adj_item_df.apply(
#             lambda row: extract_adj_item_info(row, adj_item_num), axis=1)
#
#         # 将每个新的 DataFrame 保存为 JSONL 文件
#         # 指定 force_ascii=False 保留 Unicode 字符
#         file_path = str(HOME_DIR / f"test_dataset_with_{adj_item_num}_adj_item.jsonl")
#         test_dataset_with_n_adj_item_df.to_json(file_path, orient="records", lines=True, force_ascii=False)
#     else:
#         pass
