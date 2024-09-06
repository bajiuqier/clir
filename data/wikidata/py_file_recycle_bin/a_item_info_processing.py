import pandas as pd
from pathlib import Path

from merge_files import merge_csv_files

HOME_DIR = Path(__file__).parent / 'query_entity_info'

# 合并所有的query-entity信息
pattern = r'full_train_query_entity_info\d+\.csv'

folder_path = str(HOME_DIR)
output_file = str(HOME_DIR / 'full_train_query_entity_info.csv')

merge_csv_files(folder_path=folder_path, output_file=output_file, pattern=pattern)

# 删除 中文和哈萨克语info 没有的数据
query_entity_info_df = pd.read_csv(output_file, encoding='utf-8')
query_entity_info_df.dropna(subset=['label_zh', 'label_kk', 'description_zh', 'description_kk'], how='any', inplace=True)
# 删除重复数据
query_entity_info_df.drop_duplicates(inplace=True)

item_info_file = str(HOME_DIR / 'item_info.csv')
query_entity_info_df.to_csv(item_info_file, index=False, encoding='utf-8')

