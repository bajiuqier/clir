'''
生成复合模型的数据
qrels: [query_id] [iteration] [doc_id] [relevance]
train.pairs: [query_id]	[doc_id]
queries.tsv: [type]  [id]  [text] type=query
documents.tsv: [type]  [id]  [text] type=doc
test.run: 
'''

from pathlib import Path
import pandas as pd
import ir_datasets
from tqdm import tqdm

HOME_DIR = Path.home() / 'Desktop' / 'clir' / 'data' / 'mydata' / 'clirmatrix_zh_kk'
CEDR_HOME_DIR = Path(__file__).parent / 'data_file'

# ---------------------- 构建 train.pairs ----------------------
# train pairs 直接把 train qrels 中的 query_id 和 doc_id 提出来就好了
# 只提出来 相关度为非 0 的最大的 前 pos_doc_num 个数据
# train_qrels_file = str(HOME_DIR / 'base_train_qrels.csv')
# train_qrels_df = pd.read_csv(train_qrels_file, encoding='utf-8')

# train_pairs_file = str(CEDR_HOME_DIR / 'train.pairs1')

# pos_doc_num = 1
# # 对每个 query_id 分组，并根据 relevance 排序
# # grouped = train_qrels_df.groupby('query_id', sort=False).apply(lambda x: x.sort_values(by='relevance', ascending=False)).reset_index(drop=True)

# # 提取非 0 的前 pos_doc_num 个行
# non_zero_top_n = train_qrels_df[train_qrels_df['relevance'] != 0].groupby('query_id').head(pos_doc_num)

# # 提取所有 relevance 为 0 的行
# zeros = train_qrels_df[train_qrels_df['relevance'] == 0]

# # 合并两个 DataFrame
# new_train_qrels_df = pd.concat([non_zero_top_n, zeros])
# new_train_qrels_df["query_id"] = new_train_qrels_df["query_id"].astype(int)
# new_train_qrels_df["relevance"] = new_train_qrels_df["relevance"].astype(int)

# # 按 query_id 分组 按 relevance 排序
# new_train_qrels_df = new_train_qrels_df.sort_values(['query_id', 'relevance'], ascending=[True, False]).reset_index(drop=True)


# # 选择需要写入文本文件的列
# columns_to_write = ['query_id', 'doc_id']

# 将选择的列写入到文本文件中
# with open(train_pairs_file, 'w', encoding='utf-8') as file:
#     for _, row in tqdm(new_train_qrels_df[columns_to_write].iterrows(), total=new_train_qrels_df.shape[0]):
#         line = ' '.join(map(str, row)) + '\n'
#         file.write(line)

# print(f"数据已写入 {train_pairs_file}")


# ---------------------- 构建 qrels ----------------------
# 添加 iteration 字段 为 train 和 test 的 qrels 分别 重构
# train_qrels_file = str(HOME_DIR / 'base_train_qrels.csv')
# train_qrels_df = pd.read_csv(train_qrels_file, encoding='utf-8')

# test_qrels_file = str(HOME_DIR / 'base_test_qrels.csv')
# test_qrels_df = pd.read_csv(test_qrels_file, encoding='utf-8')

# train_qrels_file = str(CEDR_HOME_DIR / 'train_qrels')
# test_qrels_file = str(CEDR_HOME_DIR / 'test_qrels')

# train_qrels_df["iteration"] = 0
# test_qrels_df["iteration"] = 0

# new_col_order = ['query_id', 'iteration', 'doc_id', 'relevance']
# train_qrels_df = train_qrels_df[new_col_order]
# test_qrels_df = test_qrels_df[new_col_order]

# with open(train_qrels_file, 'w', encoding='utf-8') as file:
#     for _, row in tqdm(train_qrels_df.iterrows(), total=train_qrels_df.shape[0]):
#         line = ' '.join(map(str, row)) + '\n'
#         file.write(line)

# print(f"数据已写入 {train_qrels_file}")

# with open(test_qrels_file, 'w', encoding='utf-8') as file:
#     for _, row in tqdm(test_qrels_df.iterrows(), total=test_qrels_df.shape[0]):
#         line = ' '.join(map(str, row)) + '\n'
#         file.write(line)

# print(f"数据已写入 {test_qrels_file}")

# ---------------------- 构建 queries.tsv ----------------------
# 加载 zh-kk clir 数据集
# CLIRMatrix_dataset_train = ir_datasets.load('clirmatrix/kk/bi139-base/zh/train')
# CLIRMatrix_dataset_test1 = ir_datasets.load('clirmatrix/kk/bi139-base/zh/test1')
# CLIRMatrix_dataset_test2 = ir_datasets.load('clirmatrix/kk/bi139-base/zh/test2')

# 加载原始查询数据
# train_queries_df = pd.DataFrame(CLIRMatrix_dataset_train.queries_iter())
# test1_queries_df = pd.DataFrame(CLIRMatrix_dataset_test1.queries_iter())
# test2_queries_df = pd.DataFrame(CLIRMatrix_dataset_test2.queries_iter())

# train_queries_df["type"] = "query"
# test1_queries_df["type"] = "query"
# test2_queries_df["type"] = "query"

# queries_df = pd.concat([train_queries_df, test1_queries_df, test2_queries_df]).reset_index(drop=True)
# queries_df.drop_duplicates(subset="query_id", keep="first", inplace=True)

# queries_df = queries_df[["type", "query_id", "text"]]

# queries_file = str(CEDR_HOME_DIR / "queries.tsv")
# queries_df.to_csv(queries_file, sep='\t', index=False, header=False)
# print(f"数据已写入 {queries_file}")


# ---------------------- 构建 documents.tsv ----------------------
# 加载 zh-kk clir 数据集
# CLIRMatrix_dataset_train = ir_datasets.load('clirmatrix/kk/bi139-base/zh/train')
# # docstore = CLIRMatrix_dataset_train.docs_store()

# # 创建一个列表来存储所有的数据
# docs_list = []
# # 提取数据
# for doc in tqdm(CLIRMatrix_dataset_train.docs_iter()):
#     docs_list.append({'type': 'doc', 'doc_id': doc.doc_id, 'text': doc.text})

# # 将数据转换成 DataFrame
# documents_df = pd.DataFrame(docs_list)
# documents_file = str(CEDR_HOME_DIR / "documents.tsv")
# # 保存 DataFrame 为 TSV 文件，不保存索引和列名
# documents_df.to_csv(documents_file, sep='\t', index=False, header=False)

# print(f"数据已保存至 {documents_file}")

# ---------------------- 构建 test.run ----------------------
# query_id iteration doc_id relevance score type
# 151 Q0 clueweb09-en0011-54-30937 1 -2.28234 run

test_qrels_file = str(HOME_DIR / 'base_test_qrels.csv')
test_qrels_df = pd.read_csv(test_qrels_file, encoding='utf-8')

test_qrels_df["iteration"] = "Q0"
test_qrels_df["score"] = 1.0
test_qrels_df["type"] = "run"

new_col_order = ['query_id', 'iteration', 'doc_id', 'relevance', 'score', 'type']
test_run_df = test_qrels_df[new_col_order]

test_run_file = str(CEDR_HOME_DIR / "test.run")

with open(test_run_file, 'w', encoding='utf-8') as file:
    for _, row in tqdm(test_run_df.iterrows(), total=test_run_df.shape[0]):
        line = ' '.join(map(str, row)) + '\n'
        file.write(line)

print(f"数据已写入 {test_run_file}")