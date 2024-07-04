from pathlib import Path
import pandas as pd

HOME_DIR = Path(__file__).parent / 'triplet_id_info'

# 去除 original_item 和 item 不相同的行
# triplet_id_filtered_cleaned_file = str(HOME_DIR / 'new_triplet_id_filtered.csv')
# triplet_id_filtered_df = pd.read_csv(triplet_id_filtered_search_results_file, encoding='utf-8')
# df_cleaned = triplet_id_filtered_df[triplet_id_filtered_df['original_item'] == triplet_id_filtered_df['item']]
# df_cleaned.to_csv(triplet_id_filtered_cleaned_file, index=False, encoding='utf-8')

# 过滤掉 item 和 adjitem 缺少中文和哈萨克语的数据

triplet_id_file = str(HOME_DIR / 'new_triplet_id_filtered.csv')
item_info_file = str(HOME_DIR.parent / 'data_file' / 'item_info.csv')
adjitem_info_file = str(HOME_DIR.parent / 'data_file' / 'adjitem_info.csv')
filtered_triplet_id_file = str(HOME_DIR / 'filtered_triplet_id.csv')

triplet_id_df = pd.read_csv(triplet_id_file, encoding='utf-8')
item_info_df = pd.read_csv(item_info_file, encoding='utf-8')
adjitem_info_df = pd.read_csv(adjitem_info_file, encoding='utf-8')

merge_df = pd.merge(triplet_id_df, item_info_df, left_on='item', right_on='item', how='left')
merge_df.dropna(subset=['label_zh', 'label_kk', 'description_zh', 'description_kk'], how='any', inplace=True)

merge_df = merge_df[['item', 'property', 'adjItem']]
merge_df = pd.merge(merge_df, adjitem_info_df, left_on='adjItem', right_on='item', how='left')
merge_df.dropna(subset=['label_zh', 'label_kk', 'description_zh', 'description_kk'], how='any', inplace=True)
# 删除重复数据
merge_df.drop_duplicates(inplace=True)
# 和 adjitem_info_df 合并后 因为两个 df 中都含有 item 所以 pd 会自动在 把item 改为 item_x item_y
merge_df = merge_df[['item_x', 'property', 'adjItem']]
merge_df.rename(columns={'item_x': 'item'}, inplace=True)
# 现在 triplet id 中所有item都有 中文和哈语的信息
merge_df.to_csv(filtered_triplet_id_file, index=False, encoding='utf-8')


# 可以不用做这一步 上面的处理 已经把 triple id 的数量 干到 6000多了
# 对每个 item 的 每个 属性 进行特定数量的提取
# triplet_id_10_file = str(HOME_DIR / 'filtered_triplet_id_10.csv')
# 按 'item' 和 'property' 分组，提取每组的前10个 'adjItem'
# grouped = merge_df.groupby(['item', 'property']).head(10)
# # 重置索引并查看新的 DataFrame
# grouped.reset_index(drop=True, inplace=True)
# grouped.to_csv(triplet_id_10_file, index=False, encoding='utf-8')