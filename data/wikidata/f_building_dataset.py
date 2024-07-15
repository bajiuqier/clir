from pathlib import Path
import ir_datasets
import pandas as pd
import jsonlines

HOME_DIR = Path(__file__).parent / 'data_file'

# mBERT_path = str(HOME_DIR.parent / 'models' / 'models--bert-base-multilingual-uncased')
adjitem_info_file = str(HOME_DIR / 'adjitem_info.csv')
item_info_file = str(HOME_DIR / 'item_info.csv')
property_info_file = str(HOME_DIR.parent / 'property_info' / 'property_info.csv')
triplet_id_file = str(HOME_DIR.parent / 'triplet_id_info' / 'filtered_triplet_id.csv')
qrels_file = str(HOME_DIR / 'qrels.csv')
query2qid_file = str(HOME_DIR / 'query2qid.csv')

train_dataset_file = str(HOME_DIR / 'train_dataset.jsonl')
test_dataset_file = str(HOME_DIR / 'test_dataset.jsonl')
train_qrels_file = str(HOME_DIR / 'train_qrels.csv')
test_qrels_file = str(HOME_DIR / 'test_qrels.csv')

# 读取CSV文件
adjitem_info = pd.read_csv(adjitem_info_file, encoding='utf-8').astype(str)
item_info = pd.read_csv(item_info_file, encoding='utf-8').astype(str)
triplet_id = pd.read_csv(triplet_id_file, encoding='utf-8').astype(str)
qrels = pd.read_csv(qrels_file, encoding='utf-8')
qrels['query_id'] = qrels['query_id'].astype(str)
qrels['doc_id'] = qrels['doc_id'].astype(str)

query2qid = pd.read_csv(query2qid_file, encoding='utf-8').astype(str)


# ----------------- 构建 训练和测试使用的 qrels -----------------
# 步骤1: 读取jsonl文件并提取query_id
train_query_ids = set()
with jsonlines.open(train_dataset_file, mode='r') as reader:
    for obj in reader:
        train_query_ids.add(obj['query_id'])

test_query_ids = set()
with jsonlines.open(test_dataset_file, mode='r') as reader:
    for obj in reader:
        test_query_ids.add(obj['query_id'])

# 步骤2: 读取qrels数据

# 步骤3: 只保留存在于jsonl文件中的query_id对应的行
train_qrels_df = qrels[qrels['query_id'].isin(train_query_ids)]
train_qrels_df.to_csv(train_qrels_file, index=False, encoding='utf-8')

test_qrels_df = qrels[qrels['query_id'].isin(test_query_ids)]
test_qrels_df.to_csv(test_qrels_file, index=False, encoding='utf-8')

# 现在qrels_df_filtered包含了只与jsonl文件中的query_id匹配的行

# ----------------- 构建 训练和测试使用的 qrels -----------------




# # 使用 ir_datasets 加载文档内容
# CLIRMatrix_dataset = ir_datasets.load('clirmatrix/kk/bi139-full/zh/train') 
# docstore = CLIRMatrix_dataset.docs_store()

# # 构建JSONL数据
# jsonl_data = []

# adjitem_num = 3
# size = int(0.8 * len(query2qid))
# train_query2qid = query2qid.iloc[:size]
# test_query2qid = query2qid.iloc[size:]

# for _, query_row in test_query2qid.iterrows():
#     query_id = query_row['query_id']
#     query_text = query_row['text']
#     query_qid = query_row['qid']
    
#     # 获取查询对应实体的信息
#     q_item = item_info[item_info['item'] == query_qid].iloc[0]

#     # 检查q_item中的'label_zh', 'label_kk' description_zh description_kk 是否有一个为空
#     if q_item[['label_zh', 'label_kk', 'description_zh', 'description_kk']].isnull().any():
#         continue

#     q_item_info = {
#         "label_zh": q_item['label_zh'],
#         "label_kk": q_item['label_kk'],
#         "description_zh": q_item['description_zh'],
#         "description_kk": q_item['description_kk']
#     }
    
#     # 获取相邻实体的信息
#     # adj_items = filtered_triplet_id[filtered_triplet_id['item'] == qid]['adjItem'].unique()
#     adj_items = triplet_id[triplet_id['item'] == query_qid]['adjItem']

#     # 舍弃相邻实体数量为0的query
#     if len(adj_items) == 0:
#         continue
#     elif len(adj_items) > 0 and len(adj_items) < adjitem_num:
#         adj_items = adj_items.tolist()
#         while len(adj_items) < adjitem_num:
#             adj_items.append(adj_items[(adjitem_num - len(adj_items)) % len(adj_items)])
#         adj_items = pd.Series(adj_items)
#     else:
#         # replace=False 是 pandas sample() 方法的一个参数，表示在抽样时不进行重复抽样
#         # sampled_adj_items = adj_items.sample(adjitem_num, replace=False)
#         adj_items = adj_items.sample(adjitem_num)

#     adj_item_info = {
#         "label_zh": [],
#         "label_kk": [],
#         "description_zh": [],
#         "description_kk": []
#     }
    
#     for adj_item in adj_items:
#         adj_info = adjitem_info[adjitem_info['item'] == adj_item].iloc[0]
#         # 这里可以判断一下 adj_item 中的下面的信息是否 为空
#         adj_item_info["label_zh"].append(adj_info['label_zh'])
#         adj_item_info["label_kk"].append(adj_info['label_kk'])
#         adj_item_info["description_zh"].append(adj_info['description_zh'])
#         adj_item_info["description_kk"].append(adj_info['description_kk'])
    

#     query_docs = qrels[qrels['query_id'] == query_id]
#     if len(query_docs) == 0:
#         continue
#     else:
#         pos_doc_ids = query_docs[query_docs['relevance'] != 0]['doc_id'][:3]
#         # neg_doc_ids = query_docs[query_docs['relevance'] == 0]['doc_id'][:3]
#         neg_doc_ids = query_docs[query_docs['relevance'] == 0]['doc_id'].sample(3)

        
#         pos_doc_texts = [docstore.get(doc_id).text for doc_id in pos_doc_ids]
#         neg_doc_texts = [docstore.get(doc_id).text for doc_id in neg_doc_ids]
    
#     # if len(pos_doc_texts) == 0:
#     #     continue
    
#     jsonl_data.append({
#         "query_id": query_id,
#         "query": query_text,
#         "pos_doc": pos_doc_texts,
#         "neg_doc": neg_doc_texts,
#         "q_item_info": q_item_info,
#         "adj_item_info": adj_item_info
#     })

# # 将数据写入JSONL文件
# with jsonlines.open(test_dataset_file, mode='w') as writer:
#     writer.write_all(jsonl_data)



'''
有这么一些数据文件：
1. adjitem_info.csv: 查询query相关实体的相邻实体信息。有 item,label_zh,label_kk,label_en,description_zh,description_kk,description_en 7列数据， 其中 item 是相邻实体的 qid
2. item_info.csv: 查询query相关实体信息，有 item,label_zh,label_kk,label_en,description_zh,description_kk,description_en 7列数据， 其中 item 是查询query相关实体的 qid
3. filtered_triplet_id.csv: 三元组数据，有 item,property,adjItem 3列数据，其中 item 是查询query相关实体的 qid， adjItem 是是相邻实体的 qid， property 对应属性id， 每个item对应多个adjItem
4. qrels.csv: 查询和相关文档信息，有 query_id,doc_id,relevance,iteration 4列数据，每个query_id 对应 100 个doc_id，relevance为query和doc的相关度，值为6， 5， 4， 3， 2， 1， 0，100个doc_id也是按照相关性程度从高到底排列的。
5. query2qid.csv: 有 query_id,text,qid 3列数据 其中 query_id 是查询id text 是查询的文本内容 qid 是查询对应的相关实体的qid
6. 另外还有一个使用 ir_datasets 加载的文档的 命名元组数据 存储着 doc_id 和文档内容

现在我想将这些数据重构为一个jsonl格式的数据
有 query pos_doc neg_doc q_item_info adj_item_info 几个数据
其中 
query 是查询的文本 字符串
pos_doc和neg_doc分别是 query对应的正样本和负样本 文档字符串 是一个列表， 如果query有多个正样本（相关度不为0的文档） 就按相关度从高到低 取出3个样本，如果只有一个正样本 就只取一个正本样本， 负样本的数量同样也是取3个
q_item_info 是一个json数据 有 label_zh、label_kk、description_zh、description_kk 分别存储 查询对应的相关实体的标签和描述的中文和哈萨克语的信息 键值是一个字符串
adj_item_info 是一个json数据 有 label_zh、label_kk、description_zh、description_kk 分别存储 当前查询对应的相关实体的n个相邻实体的中文和哈萨克语的信息 键值是一个列表，列表的长度就是相邻实体的个数 n

如果相邻实体的数量为0 就直接舍弃这条jsonl数据 即舍弃当前query的所有数据。
如果相邻实体的数量不足n个 就重复现有的相邻实体 使其数量达到 n个
'''

