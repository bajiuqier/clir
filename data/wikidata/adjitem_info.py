import requests
import pandas as pd
import ir_datasets
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# from get_query_wikidataID import process_query
from get_item_info import get_item_info
# from data_processing import filter_qid

HOME_DIR = Path(__file__).parent / 'data_file_2'

adjitem_file = str(HOME_DIR / 'adjItem_10.csv')
# property_info_file = str(HOME_DIR / 'property_info.csv')
# property_info_kk_file = str(HOME_DIR / 'property_info_kk.csv')


adjitem_df = pd.read_csv(adjitem_file, encoding='utf-8')
unique_adj_items = adjitem_df['adjItem'].drop_duplicates().tolist()
# unique_properties = adjitem_df['property'].drop_duplicates().tolist()

batch_size = 10000

# for index in range(30000, len(unique_adj_items), batch_size):
index = 30000

info_data = []
with ThreadPoolExecutor(max_workers=16) as executor:
    future_to_item = {executor.submit(get_item_info, item_id): item_id for item_id in unique_adj_items[index:index + batch_size]}
    for future in tqdm(as_completed(future_to_item), total=len(unique_adj_items[index:index + batch_size])):
        info_data.append(future.result())

info_df = pd.DataFrame(info_data)
adjitem_info_file = str(HOME_DIR / f'adjitem_info{index}.csv')
info_df.to_csv(adjitem_info_file, index=False, encoding='utf-8')

adjitem_info_KK_file = str(HOME_DIR / f'adjitem_info_kk{index}.csv')

# 删除 label_kk description_kk 为空的数据
info_df = info_df.dropna(subset=['label_kk', 'description_kk'], how='any')
info_df.to_csv(adjitem_info_KK_file, index=False, encoding='utf-8')