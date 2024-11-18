from pathlib import Path
import jsonlines
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import random

HOME_DIR = Path(__file__).parent / 'data_file'

train_dataset_file_path = str(HOME_DIR / 'train_dataset.jsonl')
test_dataset_file_path = str(HOME_DIR / 'test_dataset.jsonl')

train_triples_file_path = str(HOME_DIR / 'base_train_triplet_id_final.csv')
train_adj_item_info_file_path = str(HOME_DIR / 'base_train_adj_item_info_filled.csv')

test_triples_file_path = str(HOME_DIR / 'base_test_triplet_id_final.csv')
test_adj_item_info_file_path = str(HOME_DIR / 'base_test_adj_item_info_filled.csv')

# train_dataset_df = pd.read_json(train_dataset_file_path, lines=True, encoding='utf-8')
# train_dataset_df['query_id'] = train_dataset_df['query_id'].astype(str)

test_dataset_df = pd.read_json(test_dataset_file_path, lines=True, encoding='utf-8')
test_dataset_df['query_id'] = test_dataset_df['query_id'].astype(str)

triples_df = pd.read_csv(test_triples_file_path, encoding='utf-8')
adj_item_info_df = pd.read_csv(test_adj_item_info_file_path, encoding='utf-8')


# 定义一个函数来提取 adj_item_info 中指定的前 n 个元素
def extract_adj_item_info(row, n):
    adj_item_info = row["adj_item_info"]
    new_adj_item_info = {k: v[:n] for k, v in adj_item_info.items()}
    return new_adj_item_info


adj_item_num = 3
print(f"生成train_dataset_with_{adj_item_num+1}_adj_item_df")


a = 0
b = 0
c = 0

for index, row in tqdm(test_dataset_df.iterrows(), total=test_dataset_df.shape[0]):
# for index, row in tqdm(train_dataset_df.iterrows(), total=train_dataset_df.shape[0]):
    existing_adj_item_info = row['adj_item_info']
    q_item_qid = row['q_item_qid']

    # 从 triples 中获取所有相邻的 adj_item_qid
    adj_item_qids = triples_df[triples_df['q_item_qid'] == q_item_qid]['adj_item_qid'].tolist()
    
    if not adj_item_qids:
        a += 1
        # print("当前 q_item_qid 对应的 adj_item_qids 为空 无法添加新的数据")

    # 获取当前 q_item_qid 所有相邻实体的信息
    adj_item_info_for_q_item_df = adj_item_info_df[adj_item_info_df['qid'].isin(adj_item_qids)]
    
    if adj_item_info_for_q_item_df.shape[0] == 0:
        b += 1
        # print("当前 q_item_qid 对应的 adj_item_info 为空 无法添加新的数据")
    
    # 根据 label_zh 的数据来依次增加相邻实体相关信息的数量
    existing_labels_zh = set(existing_adj_item_info.get('label_zh'))
    # 得到新的 label_zh 列表 用来添加新的相邻实体信息
    non_duplicate_adj_item_info = list(
        set(adj_item_info_for_q_item_df.get('label_zh', pd.Series(dtype=object)).to_list()) - existing_labels_zh
        )
    
    if not non_duplicate_adj_item_info:
        c += 1
        # print("当前 q_item_qid 对应的所有相邻实体的 label_zh 数据集合与现在训练数据的 label_zh 数据集合相等 无法添加新的数据")
    
    # 已经存在的非重复的相邻实体信息数量
    the_num_of_non_duplicate_existing_adj_item_info = len(existing_labels_zh)
    
    # if not adj_item_qids or not non_duplicate_adj_item_info:
    if adj_item_info_for_q_item_df.shape[0] == 0 or not non_duplicate_adj_item_info:
        
        # print("当前 q_item_qid 对应的相邻实体信息为空 无法添加新的数据")
        # print("或者")
        # print("当前 q_item_qid 对应的所有相邻实体的 label_zh 数据集合与现在训练数据的 label_zh 数据集合相等 无法添加新的数据")
        
        row['adj_item_info']['label_zh'].append(
            row['adj_item_info']['label_zh'][adj_item_num % the_num_of_non_duplicate_existing_adj_item_info]
            )
        row['adj_item_info']['label_kk'].append(
            row['adj_item_info']['label_kk'][adj_item_num % the_num_of_non_duplicate_existing_adj_item_info]
            )
        row['adj_item_info']['description_zh'].append(
            row['adj_item_info']['description_zh'][adj_item_num % the_num_of_non_duplicate_existing_adj_item_info]
            )
        row['adj_item_info']['description_kk'].append(
            row['adj_item_info']['description_kk'][adj_item_num % the_num_of_non_duplicate_existing_adj_item_info]
            )
    else:
        
        # 根据得到的新的 label_zh 列表 来获取 新的相邻实体信息 并随机选择 1 个相邻实体
        adj_item_info_adding: pd.Series = \
        adj_item_info_for_q_item_df[adj_item_info_for_q_item_df['label_zh'].isin(non_duplicate_adj_item_info)].sample(1).iloc[0]
        
        row['adj_item_info']['label_zh'].append(adj_item_info_adding['label_zh'])
        row['adj_item_info']['label_kk'].append(adj_item_info_adding['label_kk'])
        row['adj_item_info']['description_zh'].append(adj_item_info_adding['description_zh'])
        row['adj_item_info']['description_kk'].append(adj_item_info_adding['description_kk'])
            


# file_path = str(HOME_DIR / f"train_dataset_with_{adj_item_num+1}_adj_item.jsonl")
# train_dataset_df.to_json(file_path, orient="records", lines=True, force_ascii=False)

file_path = str(HOME_DIR / f"test_dataset_with_{adj_item_num+1}_adj_item.jsonl")
test_dataset_df.to_json(file_path, orient="records", lines=True, force_ascii=False)

print(f"数据处理完成 保存路径为 {file_path}")

print(a, b, c)


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
