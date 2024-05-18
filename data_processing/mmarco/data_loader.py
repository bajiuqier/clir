import ir_datasets
import pandas as pd
import random
import pathlib
from pathlib import Path

'''
从ir_datasets中加载 mmarco 数据集， 构建随机构建少量的跨语言数据。
其中负样本是随机构建的。

改方法也被抛弃。使用 mmarco 来构建跨语言数据，请阅读 CLData_building 内容。
'''
current_file_path = Path(__file__).resolve()
trip_train_data_path = str(current_file_path.parent / 'trip_train_data.csv')



# 从ir_datasets加载 查询语言 和 文档语言 数据
mmarco_v2_zh_train = ir_datasets.load("mmarco/v2/zh/train")
mmarco_v2_ru_train = ir_datasets.load("mmarco/v2/ru/train")

# 加载 查询语言的 qrels 数据
mmarco_v2_zh_train_qrels_df = pd.DataFrame(mmarco_v2_zh_train.qrels_iter())
# mmarco_v2_ru_train_qrels_df = pd.DataFrame(mmarco_v2_ru_train.qrels_iter())

# 随机取出 5000 行数据 
# 使用random_state以确保结果可重复
mmarco_v2_zh_ru_train_5k_qrels_df = mmarco_v2_zh_train_qrels_df.sample(n=5000, random_state=12).reset_index(drop=True)
# 去重的 操作 这里不做 去重  使用 数据集的原生数据
# mmarco_v2_zh_ru_train_5k_qrels_df = mmarco_v2_zh_ru_train_5k_qrels_df.drop_duplicates(subset='doc_id', keep='first').reset_index(drop=True)

# 取出查询数据 dataframe 类型
mmarco_v2_zh_train_queries_df = pd.DataFrame(mmarco_v2_zh_train.queries_iter())
# doc 容器
mmarco_v2_ru_train_docstore = mmarco_v2_ru_train.docs_store()

# 将查询文本数据和 qrels 数据进行合并
queries_train_data_df = mmarco_v2_zh_ru_train_5k_qrels_df.merge(
    mmarco_v2_zh_train_queries_df,
    on='query_id',
    how='inner'
)
# 删除 暂时没用的列数据
# queries_train_data_df.drop(['relevance', 'iteration'], axis=1, inplace=True)
# 重命名 query 的 “text” 列名
queries_train_data_df.rename(columns={'text': 'query'}, inplace=True)

# 取出正样本 doc_id
pos_doc_ids = mmarco_v2_zh_ru_train_5k_qrels_df['doc_id'].astype(str).tolist()
# 构建 负样本 doc_id
neg_doc_ids = pos_doc_ids[:]
random.shuffle(neg_doc_ids)
# 确保 每一个正样本 对应的负样本与本身不同
for i in range(len(pos_doc_ids)):
    while neg_doc_ids[i] == pos_doc_ids[i]:
        random.shuffle(neg_doc_ids)

# 构建 pos样本 和 neg样本 列表数据
docs_train_data_list = []
for index, row in mmarco_v2_zh_ru_train_5k_qrels_df.iterrows():
    docs_train_data_list.append([row['query_id'],
                 row['doc_id'],
                 mmarco_v2_ru_train_docstore.get(pos_doc_ids[index]).text,
                 mmarco_v2_ru_train_docstore.get(neg_doc_ids[index]).text
                 ])
'''
其实也可以在 上面的列表数据中 去 append query文本
mmarco_v2_zh_train_queries_df.loc[mmarco_v2_zh_train_queries_df['query_id'] == row['query_id'], 'text'].values[0]
但是 这样操作 会让 处理速度变慢很多
'''

docs_train_data_df = pd.DataFrame(docs_train_data_list, columns=['query_id', 'doc_id', 'positive', 'negative'])
# 将查询文本数据 和文档文本数据 进行合并
trip_train_data_df = queries_train_data_df.merge(docs_train_data_df, on=['query_id', 'doc_id'], how='inner')
# trip_train_data_no_id_df = 
# 导出 csv 文件
trip_train_data_df.to_csv(trip_train_data_path, index=False)