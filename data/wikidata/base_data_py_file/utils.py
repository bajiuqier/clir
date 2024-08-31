import string
from pathlib import Path
import pandas as pd
import os
import glob
import re

def is_english_or_chinese(text):
    '''
    检查字符串是中文还是英文还是混合两种语言
    全为中文返回 字符串 "Chinese"
    全为英文返回 字符串 "English"
    混合两种语言返回 字符串 "Mixed"
    '''

    # 删除所有空格符
    text = text.replace(' ', '')
    # 去除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 初始化计数器
    english_count = 0
    chinese_count = 0
    
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            chinese_count += 1
        elif '\u0041' <= char <= '\u005a' or '\u0061' <= char <= '\u007a':
            english_count += 1
    
    if chinese_count > 0 and english_count == 0:
        return "Chinese"
    elif english_count > 0 and chinese_count == 0:
        return "English"
    else:
        return "Mixed"
    

def merge_csv_files(folder_path, output_file, pattern):
    
    # 获取所有匹配的文件
    # all_files = glob.glob(os.path.join(folder_path, 'full_train_QID_filtered_search_results*.csv'))
    all_files = glob.glob(os.path.join(folder_path, '*.csv'))

    
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


if __name__ == "__main__":

    pass