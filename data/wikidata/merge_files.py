from pathlib import Path
import pandas as pd
import os
import glob
import re



def merge_csv_files(folder_path, output_file):
    # 定义文件模式
    pattern = r'adjitem_info_kk\d+\.csv'
    
    # 获取所有匹配的文件
    all_files = glob.glob(os.path.join(folder_path, 'adjitem_info_kk*.csv'))
    
    # 筛选符合模式的文件
    matched_files = [f for f in all_files if re.match(pattern, os.path.basename(f))]
    
    # 按文件名中的数字排序
    matched_files.sort(key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))
    
    # 读取并合并所有CSV文件
    df_list = [pd.read_csv(csv_file, encoding='utf-8') for csv_file in matched_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # 将合并后的数据保存为新的CSV文件
    combined_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"合并完成，输出文件: {output_file}")

HOME_DIR = Path(__file__).parent / 'data_file_2'

# 使用示例
folder_path = str(HOME_DIR)  # 替换为您的文件夹路径
output_file =str(HOME_DIR / 'merged_kk_adjitem_info.csv')   # 输出文件名

merge_csv_files(folder_path, output_file)