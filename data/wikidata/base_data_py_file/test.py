import pandas as pd
from pathlib import Path

HOME_DIR = Path(__file__).parent.parent / 'base_data_file'
triplet_id_file = HOME_DIR / 'base_train_triplet_id_filtered.csv'

# 读取 CSV 文件
df = pd.read_csv(str(HOME_DIR / 'base_train_triplet_id_filtered.csv'), encoding='utf-8')

# 重命名列
df.rename(columns={'item_qid': 'q_item_qid'}, inplace=True)

# 保存修改后的 CSV 文件
df.to_csv(triplet_id_file, index=False, encoding='utf-8')

