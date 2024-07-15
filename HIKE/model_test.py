from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, NamedTuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import ir_datasets

HOME_DIR = Path(__file__).parent
DATA_HOME_DIR = Path(__file__).parent.parent / 'data' / 'wikidata' / 'data_file'

mBERT_path = str(HOME_DIR.parent / 'models' / 'models--bert-base-multilingual-uncased')
adjitem_info_file = str(DATA_HOME_DIR / 'adjitem_info.csv')
item_info_file = str(DATA_HOME_DIR / 'item_info.csv')
property_info_file = str(DATA_HOME_DIR.parent / 'property_info' / 'property_info.csv')
triplet_id_file = str(DATA_HOME_DIR.parent / 'triplet_id_info' / 'filtered_triplet_id.csv')
qrels_file = str(DATA_HOME_DIR / 'qrels.csv')
query2qid_file = str(DATA_HOME_DIR / 'query2qid.csv')

class MyDataset(Dataset):
    def __init__(self, adjitem_num: int=3):
        super().__init__()

        self.CLIRMatrix_dataset = ir_datasets.load('clirmatrix/kk/bi139-full/zh/train') 
        self.docstore = self.CLIRMatrix_dataset.docs_store()
        self.query2qid_df = pd.read_csv(query2qid_file, encoding='utf-8').astype(str)
        self.qrels_df = pd.read_csv(qrels_file, encoding='utf-8')
        self.qrels_df['query_id'] = self.qrels_df['query_id'].astype(str)
        self.qrels_df['doc_id'] = self.qrels_df['doc_id'].astype(str)
        # self.qrels_df['relevance'] = self.qrels_df['relevance'].astype(int)

        self.item_info = pd.read_csv(item_info_file, encoding='utf-8').astype(str)
        self.adjitem_info = pd.read_csv(adjitem_info_file, encoding='utf-8').astype(str)
        self.triplet_id = pd.read_csv(triplet_id_file, encoding='utf-8').astype(str)
        # self.triplet_id[]
        self.adjitem_num = adjitem_num
        self.dataset = self.get_data(
            queries=self.query2qid_df,
            docstore=self.docstore,
            qrels=self.qrels_df,
            item_info=self.item_info,
            triplet_id=self.triplet_id,
            adjitem_info=self.adjitem_info,
            num=self.adjitem_num
        )

    def __len__(self):
        return len(self.qrels_df.groupby('query_id'))
    
    def __getitem__(self, index):
        # item = self.dataset[]
        # return qd, entity
        pass

    @staticmethod
    def get_data(
        queries: pd.DataFrame,
        docstore: NamedTuple,
        qrels: pd.DataFrame,
        item_info: pd.DataFrame,
        triplet_id: pd.DataFrame,
        adjitem_info: pd.DataFrame,
        num: int=3) -> List[list[str]]:
        qd = []
        q_item_label_s = []
        q_item_label_t = []
        q_item_desc_s = []
        q_item_desc_t = []
        adj_item_label_s = []
        adj_item_label_t = []
        adj_item_desc_s = []
        adj_item_desc_t = []

        for query_id, group in qrels.groupby('query_id'):
            query = queries[queries['query_id'] == query_id]['text'].values[0]            
            # 取出和当前 query 最相关的一个 document 作为 正样本
            pos_doc_id = group[group['relevance'] != 0]['doc_id'].values[0]
            pos_doc = docstore.get(pos_doc_id).text
            # 随机取出一个相关度为 0  的 document 作为 负样本
            neg_doc_id = group[group['relevance'] == 0]['doc_id'].sample(1).values[0] if not group[group['relevance'] == 0].empty else None
            neg_doc = docstore.get(neg_doc_id).text if neg_doc_id else None

            if neg_doc:  # 确保负样本存在
                qd.append([query, pos_doc, neg_doc])

            query_qid = queries[queries['query_id'] == query_id]['qid'].values[0]
            q_item_label_s.append(item_info[item_info['item'] == query_qid]['label_zh'].values[0])
            q_item_label_t.append(item_info[item_info['item'] == query_qid]['label_kk'].values[0])
            q_item_desc_s.append(item_info[item_info['item'] == query_qid]['description_zh'].values[0])
            q_item_desc_t.append(item_info[item_info['item'] == query_qid]['description_kk'].values[0])


            adj_items = triplet_id[triplet_id['item'] == query_qid]['adjItem']
            if len(adj_items) < num:
                adj_items = adj_items.tolist()
                while len(adj_items) < num: 
                    adj_items.append(adj_items[(num - len(adj_items)) % len(adj_items)])
                adj_items = pd.Series(adj_items)

            # replace=False 是 pandas sample() 方法的一个参数，表示在抽样时不进行重复抽样
            sampled_adj_items = adj_items.sample(num, replace=False)

            label_s = []
            label_t = []
            desc_s = []
            desc_t = []
            for adj_item_qid in sampled_adj_items:
                label_s.append(adjitem_info[adjitem_info['item'] == adj_item_qid]['label_zh'].values[0])
                label_t.append(adjitem_info[adjitem_info['item'] == adj_item_qid]['label_kk'].values[0])
                desc_s.append(adjitem_info[adjitem_info['item'] == adj_item_qid]['description_zh'].values[0])
                desc_t.append(adjitem_info[adjitem_info['item'] == adj_item_qid]['description_kk'].values[0])
            adj_item_label_s.append(label_s)
            adj_item_label_t.append(label_t)
            adj_item_desc_t.append(desc_s)
            adj_item_desc_t.append(desc_t)
            
        q_item_info = {'label': [q_item_label_s, q_item_label_t], 'description': [q_item_desc_s, q_item_desc_t]}
        adj_item_info = {'label': [adj_item_label_s, adj_item_label_t], 'description': [adj_item_desc_s, adj_item_desc_t]}

        return qd, q_item_info, adj_item_info
    


train_dataset = MyDataset().dataset

print(train_dataset)

# mBERT_model = BertModel.from_pretrained(mBERT_path)
# mBERT_tokenizer = BertTokenizer.from_pretrained(mBERT_path)


# query = '中国'
# doc = 'Таужыныстармен бірге жаралған шашыранды органикалық заттардың генетикалық түрі және катагендік өзгерістерге ұшырау ережесі тұрғысынан дараланатын топтамалары. Шашыранды органикалық заттардың генетикалық түрі сол шашыранды заттардан таужыныстарға қышқылдармен (HC1 және HF) әсер ету нәтижесінде бөлініп алынған және одан таушайыр тектестері мен гумин қышқылдарын аластағаннан кейінгі калған көмірпетрографиялық және химиялық сипаттарымен анықталады. Мұнай және газ геологиясы танымдық және кәсіптік-технологиялық терминдерінің түсіндірме сөздігі. Анықтамалық басылым. — Алматы: 2003 жыл. ISBN 9965-472-27-0'

# encode_pair = mBERT_tokenizer(
#     query,
#     doc,
#     padding=True,
#     truncation=True,
#     max_length=128,
#     return_tensors="pt",
# )


# print(encode_pair)
