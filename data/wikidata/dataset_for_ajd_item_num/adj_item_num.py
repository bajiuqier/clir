from pathlib import Path
import jsonlines
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

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


for adj_item_num in tqdm(range(1, 8)):
    if adj_item_num <= 3:
        train_dataset_with_n_adj_item_df = train_dataset_df.copy(deep=True)
        train_dataset_with_n_adj_item_df["adj_item_info"] = train_dataset_with_n_adj_item_df.apply(
            lambda row: extract_adj_item_info(row, adj_item_num), axis=1)

        # 将每个新的 DataFrame 保存为 JSONL 文件
        # 指定 force_ascii=False 保留 Unicode 字符
        file_path = str(HOME_DIR / f"train_dataset_with_{adj_item_num}_adj_item.jsonl")
        train_dataset_with_n_adj_item_df.to_json(file_path, orient="records", lines=True, force_ascii=False)
    else:
        pass
