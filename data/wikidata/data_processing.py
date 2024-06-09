import pandas as pd
from pathlib import Path

HOME_DIR = Path(__file__).parent
search_results_file = str(HOME_DIR / 'test1_QID_search_results.csv')
test1_QID_filtered_search_results_file = str(HOME_DIR / 'test1_QID_filtered_search_results.csv')


df = pd.read_csv(search_results_file, encoding='utf-8')

columns_to_keep = ['query', 'search_term', 'id', 'label', 'description']
# 只取 columns_to_keep 列数据，然后去除 id 列 为NaN的行数据
df_filtered = df[columns_to_keep].dropna(subset=['id'])
# 保存文件
df_filtered.to_csv(test1_QID_filtered_search_results_file, index=False, encoding='utf-8')
