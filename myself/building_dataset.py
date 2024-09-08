from pathlib import Path
import ir_datasets
import pandas as pd
import jsonlines

HOME_DIR = Path(__file__).parent / 'data'

# mBERT_path = str(HOME_DIR.parent / 'models' / 'models--bert-base-multilingual-uncased')
adjitem_info_file = str(HOME_DIR / 'adjitem_info.csv')
item_info_file = str(HOME_DIR / 'item_info.csv')
property_info_file = str(HOME_DIR / 'property_info.csv')
triplet_id_file = str(HOME_DIR / 'filtered_triplet_id.csv')
qrels_file = str(HOME_DIR / 'qrels.csv')
query2qid_file = str(HOME_DIR / 'query2qid.csv')

dataset_file = str(HOME_DIR / 'dataset.jsonl')

# 读取CSV文件
adjitem_info = pd.read_csv(adjitem_info_file, encoding='utf-8').astype(str)
item_info = pd.read_csv(item_info_file, encoding='utf-8').astype(str)
triplet_id = pd.read_csv(triplet_id_file, encoding='utf-8').astype(str)
qrels = pd.read_csv(qrels_file, encoding='utf-8')
query2qid = pd.read_csv(query2qid_file, encoding='utf-8').astype(str)

CLIRMatrix_dataset = ir_datasets.load('clirmatrix/kk/bi139-full/zh/train') 
docstore = CLIRMatrix_dataset.docs_store()
# 使用 ir_datasets 加载文档内容
# import ir_datasets
# dataset = ir_datasets.load('msmarco-passage')  # 根据你的实际数据集名称调整

# 创建一个字典来存储文档内容
doc_content = {}
for doc in dataset.docs_iter():
    doc_content[doc.doc_id] = doc.text

# 构建JSONL数据
jsonl_data = []

adjitem_num = 3

for _, query_row in query2qid.iterrows():
    query_id = query_row['query_id']
    query_text = query_row['text']
    query_qid = query_row['qid']
    
    # 获取查询对应实体的信息
    q_item = item_info[item_info['item'] == query_qid].iloc[0]
    q_item_info = {
        "label_zh": q_item['label_zh'],
        "label_kk": q_item['label_kk'],
        "description_zh": q_item['description_zh'],
        "description_kk": q_item['description_kk']
    }
    
    # 获取相邻实体的信息
    # adj_items = filtered_triplet_id[filtered_triplet_id['item'] == qid]['adjItem'].unique()
    adj_items = triplet_id[triplet_id['item'] == query_qid]['adjItem']

    # 舍弃相邻实体数量为0的query
    if len(adj_items) == 0:
        continue

    if len(adj_items) < adjitem_num:
        adj_items = adj_items.tolist()
        while len(adj_items) < adjitem_num:
            adj_items.append(adj_items[(adjitem_num - len(adj_items)) % len(adj_items)])
        adj_items = pd.Series(adj_items)

    # replace=False 是 pandas sample() 方法的一个参数，表示在抽样时不进行重复抽样
    sampled_adj_items = adj_items.sample(num, replace=False)


    adj_item_info = {
        "label_zh": [],
        "label_kk": [],
        "description_zh": [],
        "description_kk": []
    }
    
    for adj_item in adj_items:
        adj_info = adjitem_info[adjitem_info['item'] == adj_item].iloc[0]
        adj_item_info["label_zh"].append(adj_info['label_zh'])
        adj_item_info["label_kk"].append(adj_info['label_kk'])
        adj_item_info["description_zh"].append(adj_info['description_zh'])
        adj_item_info["description_kk"].append(adj_info['description_kk'])
    
    # 舍弃相邻实体数量为0的query
    if len(adj_items) == 0:
        continue
    
    # 如果相邻实体数量不足n个，重复现有的相邻实体
    n = 3
    while len(adj_item_info["label_zh"]) < n:
        adj_item_info["label_zh"].extend(adj_item_info["label_zh"][:n - len(adj_item_info["label_zh"])])
        adj_item_info["label_kk"].extend(adj_item_info["label_kk"][:n - len(adj_item_info["label_kk"])])
        adj_item_info["description_zh"].extend(adj_item_info["description_zh"][:n - len(adj_item_info["description_zh"])])
        adj_item_info["description_kk"].extend(adj_item_info["description_kk"][:n - len(adj_item_info["description_kk"])])
    
    # 获取正样本和负样本
    query_docs = qrels[qrels['query_id'] == query_id]
    pos_docs = query_docs[query_docs['relevance'] > 0]['doc_id'][:3]
    neg_docs = query_docs[query_docs['relevance'] == 0]['doc_id'][:3]
    
    pos_doc_texts = [doc_content[doc_id] for doc_id in pos_docs if doc_id in doc_content]
    neg_doc_texts = [doc_content[doc_id] for doc_id in neg_docs if doc_id in doc_content]
    
    if len(pos_doc_texts) == 0:
        continue
    
    jsonl_data.append({
        "query": query_text,
        "pos_doc": pos_doc_texts,
        "neg_doc": neg_doc_texts,
        "q_item_info": q_item_info,
        "adj_item_info": adj_item_info
    })

# 将数据写入JSONL文件
with jsonlines.open('output.jsonl', mode='w') as writer:
    writer.write_all(jsonl_data)



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
如果相邻实体的数量不足n个， 就重复现有的相邻实体 使其数量达到 n个
'''

