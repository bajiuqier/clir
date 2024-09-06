from pathlib import Path
import pandas as pd

from merge_files import merge_csv_files

HOME_DIR = Path(__file__).parent / 'adjitem_info'

# 合并 adjitem info
pattern = r'adjitem_info\d+\.csv'

folder_path = str(HOME_DIR)
output_file = str(HOME_DIR / 'all_adjitem_info.csv')

merge_csv_files(folder_path=folder_path, output_file=output_file, pattern=pattern)

# 删除 中文和哈萨克语info 没有的数据
all_adjitem_info_df = pd.read_csv(output_file, encoding='utf-8')
all_adjitem_info_df.dropna(subset=['label_zh', 'label_kk', 'description_zh', 'description_kk'], how='any', inplace=True)
# 删除重复数据
all_adjitem_info_df.drop_duplicates(inplace=True)

adjitem_info_file = str(HOME_DIR / 'adjitem_info.csv')
all_adjitem_info_df.to_csv(adjitem_info_file, index=False, encoding='utf-8')