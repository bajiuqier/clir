import pandas as pd
from pathlib import Path

HOME_DIR = Path(__file__).parent
test1_QID_search_results_file = str(HOME_DIR / 'test1_QID_search_results.csv')
test1_QID_filtered_search_results_file = str(HOME_DIR / 'test1_QID_filtered_search_results.csv')


test1_QID_df = pd.read_csv(test1_QID_search_results_file, encoding='utf-8')

columns_to_keep = ['query', 'search_term', 'id', 'label', 'description']
# 只取 columns_to_keep 列数据，然后去除 id 列 为NaN的行数据
test1_QID_filtered_df = test1_QID_df[columns_to_keep].dropna(subset=['id'])
# 保存文件
test1_QID_filtered_df.to_csv(test1_QID_filtered_search_results_file, index=False, encoding='utf-8')


# # 使用正则表达式过滤符合条件的行 匹配以 "Q" 开头后跟数字的字符串
# # 前提是 AdjItem 列 的值是 str 类型的
# df_filtered = df[df['AdjItem'].str.match(r'^Q\d+$')]
# print(df_filtered)