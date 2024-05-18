import ir_datasets
import pandas as pd
from pathlib import Path
import json
import numpy as np
import os


path_home = Path.home() / "Desktop" / "Datasets" / "CLIRMatrix"

# dataset1 = ir_datasets.load('clirmatrix/zh/bi139-full/kk/test1')
dataset2 = ir_datasets.load('clirmatrix/kk/bi139-full/zh/test1')
# dataset3 = ir_datasets.load('clirmatrix/kk/bi139-full/zh/test2')

# docstore2 = dataset2.docs_store()
queries_df2 = pd.DataFrame(dataset2.queries_iter())
qrels_df2 = pd.DataFrame(dataset2.qrels_iter())

# queries_df3 = pd.DataFrame(dataset3.queries_iter())
# qrels_df3 = pd.DataFrame(dataset3.qrels_iter())


# 每 10000 条数据 存储为一个 xlsx 然后使用 yandex translate 翻译这个 xlsx 文件
# queries_df2.loc[80000:89999].to_excel(str(path_home / 'queries_zh.kk_80000_89999.xlsx'), index=False)

# ---------------------- 合并翻译后的 queries ----------------------
# # 文件夹路径
# folder_path = str(path_home / 'BI-139' / 'queries_test1_zh_kk')
# # 存储合并后数据的DataFrame
# all_data = pd.DataFrame()

# # 遍历文件夹中的文件
# for filename in os.listdir(folder_path):

#     if filename.endswith('.xlsx') and 'queries' in filename:
#         # 构建完整的文件路径
#         file_path = os.path.join(folder_path, filename)
        
#         # 读取Excel文件，只选取'query_id'和'text'两列
#         temp_df = pd.read_excel(file_path, usecols=['query_id', 'text'])
        
#         # 将读取的数据追加到all_data DataFrame中
#         all_data = pd.concat([all_data, temp_df], ignore_index=True)

# # 将'query_id'和'text'列转换为字符串类型
# all_data[['query_id', 'text']] = all_data[['query_id', 'text']].astype(str)

# # 保存到新的Excel文件
# all_data.to_excel(str(path_home / 'BI-139' / 'queries_zh.kk.xlsx'), index=False)

# print("合并并转换完成，结果已保存至queries_zh.kk.xlsx")

# ---------------------- 合并翻译后的 queries ----------------------

# ---------------------- 验证合并后的 queries 是否相同 ----------------------


queries_translate = pd.read_excel(str(path_home / 'BI-139' / 'queries_zh.kk.xlsx'))
# 需要把 query_id 转成 str 类型
queries_translate['query_id'] = queries_translate['query_id'].astype(str)

assert queries_df2.shape[0] == queries_translate.shape[0], '翻译后的数据与原始数据的 数量不相同'

if queries_df2['query_id'].equals(queries_translate['query_id']):
    print('翻译后的数据数量和query_id与原数据 完全相同')
else:
    print('翻译后的数据数量与原数据相同  但是 query_id 存在不同')


# ---------------------- 验证合并后的 queries 是否相同 ----------------------
