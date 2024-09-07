import pandas as pd
from pathlib import Path

HOME_DIR = Path(__file__).parent

mBERT_path = str(HOME_DIR.parent / 'models' / 'models--bert-base-multilingual-uncased')
adjitem_info_file = str(HOME_DIR / 'data' / 'merged_kk_adjitem_info.csv')
item_info_file = str(HOME_DIR / 'data' / 'merged_kk_item_info.csv')
property_info_file = str(HOME_DIR / 'data' / 'property_info.csv')
triplet_id_file = str(HOME_DIR / 'data' / 'triplet_id.csv')
new_qrels_file = str(HOME_DIR / 'data' / 'new_qrels.csv')
query2qid_file = str(HOME_DIR / 'data' / 'new_qrels.csv')

new_item_info_file = str(HOME_DIR / 'data' / 'item_info.csv')


triplet_id_df = pd.read_csv(triplet_id_file, encoding='utf-8')
item_info_df = pd.read_csv(item_info_file, encoding='utf-8')

# 删除  值为空的行数据
item_info_df.dropna(subset=['label_zh', 'label_kk', 'description_zh', 'description_kk'], how='any', inplace=True)
# 删除重复值
item_info_df.drop_duplicates(inplace=True)
item_info_df.to_csv(new_item_info_file, index=False, encoding='utf-8')


merged_df = triplet_id_df.merge(item_info_df, on='item', how='left', indicator=True)
df1_cleaned = merged_df[merged_df['_merge'] != 'left_only'].drop(columns=['_merge'])

triplet_id_df = df1_cleaned[['item', 'property', 'adjitem']]

adjitem_info_df = pd.read_csv(adjitem_info_file, encoding='utf-8')
# 删除  值为空的行数据
adjitem_info_df.dropna(subset=['label_zh', 'label_kk', 'description_zh', 'description_kk'], how='any', inplace=True)
# 删除重复值
adjitem_info_df.drop_duplicates(inplace=True)

merged_df = pd.merge(triplet_id_df, item_info_df, left_on='adjitem', right_on='item', how='left', indicator=True)
df2_cleaned = merged_df[merged_df['_merge'] != 'left_only'].drop(columns=['_merge'])
triplet_id_df = df2_cleaned[['item', 'property', 'adjitem']]

qrels_df = pd.read_csv(new_qrels_file, encoding='utf-8')



print(item_info_df)
