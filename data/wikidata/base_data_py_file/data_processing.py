import pandas as pd
from pathlib import Path

HOME_DIR = Path(__file__).parent / 'data_file_2'

# --------------------- 过滤 query 的 QID ---------------------
# QID_search_results_file = str(HOME_DIR / 'full_train_QID_search_results1.csv')
# QID_filtered_search_results_file = str(HOME_DIR / 'full_train_QID_filtered_search_results1.csv')

def filter_qid(original_file: str, filtered_file: str):

    QID_df = pd.read_csv(original_file, encoding='utf-8')

    columns_to_keep = ['query', 'search_term', 'id', 'label', 'description']
    # 只取 columns_to_keep 列数据，然后去除 id 列 为NaN的行数据
    QID_filtered_df = QID_df[columns_to_keep].dropna(subset=['id'])
    # 保存文件
    QID_filtered_df.to_csv(filtered_file, index=False, encoding='utf-8')

# filter_qid(QID_search_results_file, QID_filtered_search_results_file)

# --------------------- 过滤 query 的 QID ---------------------

# --------------------- 过滤三元组 （实体-关系-实体) ---------------------
# triplet_id_search_results_file = str(HOME_DIR / 'triplet_id_search_results.csv')
# triplet_id_filtered_search_results_file = str(HOME_DIR / 'triplet_id_filtered_search_results.csv')

def filter_triplet_id(original_file: str, filtered_file: str):
    # 读取 CSV 文件
    triplet_id_df = pd.read_csv(original_file, encoding='utf-8')

    # 删除含有任何 NaN 值的行
    triplet_id_df = triplet_id_df.dropna()

    # 确保 AdjItem 列中的值是字符串类型，并且填充 NaN 值
    triplet_id_df['adjItem'] = triplet_id_df['adjItem'].astype(str)

    # 使用正则表达式过滤符合条件的行 匹配以 "Q" 开头后跟数字的字符串
    # na=False 确保 NaN 值不会引起错误。
    triplet_id_filtered_df = triplet_id_df[triplet_id_df['adjItem'].str.match(r'^Q\d+$', na=False)]

    # 将结果保存到 CSV 文件
    triplet_id_filtered_df.to_csv(filtered_file, index=False, encoding='utf-8')

# filter_triplet_id(triplet_id_search_results_file, triplet_id_filtered_search_results_file)

# 去除 original_item 和 item 不相同的行
# triplet_id_filtered_cleaned_file = str(HOME_DIR / 'new_triplet_id_filtered.csv')
# triplet_id_filtered_df = pd.read_csv(triplet_id_filtered_search_results_file, encoding='utf-8')
# df_cleaned = triplet_id_filtered_df[triplet_id_filtered_df['original_item'] == triplet_id_filtered_df['item']]

# df_cleaned.to_csv(triplet_id_filtered_cleaned_file, index=False, encoding='utf-8')

# 对每个 item 的 每个 属性 进行特定数量的提取
# adjItem_10_file = str(HOME_DIR / 'adjItem_10.csv')
# new_triplet_id_filtered_file = str(HOME_DIR / 'new_triplet_id_filtered.csv')
# new_triplet_id_filtered_df = pd.read_csv(new_triplet_id_filtered_file, encoding='utf-8')
# # 按 'item' 和 'property' 分组，提取每组的前10个 'adjItem'
# grouped = new_triplet_id_filtered_df.groupby(['item', 'property']).head(10)
# # 重置索引并查看新的 DataFrame
# df_new = grouped.reset_index(drop=True)
# df_new.to_csv(adjItem_10_file, index=False, encoding='utf-8')


# 检查 是否有 空 NaN 值
# test1_triplet_id_filtered_df = pd.read_csv(test1_triplet_id_filtered_search_results_file, encoding='utf-8')
# print(test1_triplet_id_filtered_df.isnull().values.any())

# --------------------- 过滤三元组 （实体-关系-实体) ---------------------


# --------------------- 删除 实体、属性英文信息为空的行数据 ---------------------
# query_entity_info_file = str(HOME_DIR / 'full_train_query_entity_info1.csv')
# query_entity_info_kk_file = str(HOME_DIR / 'full_train_query_entity_info1_0.csv')
# query_entity_info_df = pd.read_csv(query_entity_info_file, encoding='utf-8')

# 删除 label_kk description_kk 为空的数据
# query_entity_info_df = query_entity_info_df.dropna(subset=['label_kk', 'description_kk'], how='any')
# query_entity_info_df.to_csv(query_entity_info_kk_file, index=False, encoding='utf-8')

# query_entity_info_1_file = str(HOME_DIR / 'query_entity_info_1.csv')

# query_adjacent_entity_info_file = str(HOME_DIR / 'query_adjacent_entity_info.csv')
# query_adjacent_entity_info_1_file = str(HOME_DIR / 'query_adjacent_entity_info_1.csv')

# property_info_file = str(HOME_DIR / 'property_info.csv')
# property_info_1_file = str(HOME_DIR / 'property_info_1.csv')

def filter_item_info(original_file: str, filtered_file: str):

    item_info_df = pd.read_csv(original_file, encoding='utf-8')

    if item_info_df['label_en'].isnull().any():
        item_info_df = item_info_df.dropna(subset=['label_en'])

    if item_info_df['description_en'].isnull().any():
        item_info_df = item_info_df.dropna(subset=['description_en'])

    item_info_df.to_csv(filtered_file, index=False, encoding='utf-8')

# --------------------- 删除 实体、属性英文信息为空的行数据 ---------------------

full_triplet_id_10_file = str(HOME_DIR / 'adjItem_10.csv')
adjitem_info_file = str(HOME_DIR / 'merged_kk_adjitem_info.csv')
triplet_info_file = str(HOME_DIR / 'triplet_info.csv')
triplet_id_file = str(HOME_DIR / 'triplet_id.csv')


# full_triplet_id_10_df = pd.read_csv(full_triplet_id_10_file, encoding='utf-8')
# adjitem_info_df = pd.read_csv(adjitem_info_file, encoding='utf-8')

# triplet_info_df = pd.merge(full_triplet_id_10_df, adjitem_info_df, how='left', left_on='adjItem', right_on='item')
# triplet_info_df.dropna(subset=['label_kk', 'description_kk'], how='any', inplace=True)
# triplet_info_df.to_csv(triplet_info_file, index=False, encoding='utf-8')
# triplet_info_df = pd.read_csv(triplet_info_file, encoding='utf-8')
# triplet_id_df = triplet_info_df[['item_x', 'property', 'adjItem']]
# triplet_id_df.to_csv(triplet_id_file, index=False, encoding='utf-8')


