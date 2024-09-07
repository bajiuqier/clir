import ir_datasets
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from argments import parse_args



class CLIRMatrixDataset(Dataset):
    def __init__(self, args: parse_args) -> None:
        super().__init__()
        self.dataset = ir_datasets.load('clirmatrix/kk/bi139-base/zh/train')
        self.queries_df = pd.DataFrame(self.dataset.queries_iter())
        self.docstore = self.dataset.docs_store()
        self.qrels_df = pd.DataFrame(self.dataset.qrels_iter())
        self.train_data = self.get_data(self.queries_df, self.docstore, self.qrels_df)

    def __len__(self):
        # return len(self.queries_df)
        return len(self.train_data)
    
    def __getitem__(self, index) -> Tuple[str, list[str]]:
        item = self.train_data[index]
        query = item[0]
        documet = []
        documet.append(item[1])
        documet.append(item[2])
        return query, documet
    
    @staticmethod
    def get_data(queries, documents, qrels) -> List[list[str]]:
        train_data = []
        for query_id , group in tqdm(qrels.groupby('query_id'), total=qrels['query_id'].nunique()):
            query = queries[queries['query_id'] == query_id]['text'].values[0]
            # 取出和当前 query 最相关的一个 document 作为 正样本
            pos_doc_id = group[group['relevance'] != 0]['doc_id'].values[0]
            pos_doc = documents.get(pos_doc_id).text
            # 随机取出一个相关度为 0  的 document 作为 负样本
            neg_doc_id = group[group['relevance'] == 0]['doc_id'].sample(1).values[0] if not group[group['relevance'] == 0].empty else None
            neg_doc = documents.get(neg_doc_id).text if neg_doc_id else None

            if neg_doc:  # 确保负样本存在
                train_data.append([query, pos_doc, neg_doc])
        return train_data
                

@dataclass
class DataCollatorForCrossEncoder(DataCollatorWithPadding):

    max_len: int = 256
    
    def __call__(self, features: List[Tuple[str, list[str]]]) -> Dict[str, Any]:

        query = [f[0] for f in features]
        document = [f[1] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(document[0], list):
            document = sum(document, [])
        # 将 query 中的内容 重复两份
        query = sum([[element]*2 for element in query], [])

        batch = self.tokenizer(
                query,
                document,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )

        return batch


class DatasetForTest(Dataset):
    def __init__(self, args: parse_args) -> None:
        super().__init__()
        self.dataset = ir_datasets.load('clirmatrix/kk/bi139-base/zh/test1')
        self.queries_df = pd.DataFrame(self.dataset.queries_iter())
        self.docstore = self.dataset.docs_store()
        self.qrels_df = pd.DataFrame(self.dataset.qrels_iter())
        # self.train_data = self.get_data(self.queries_df, self.docstore, self.qrels_df)
    def __len__(self):
        return self.qrels_df.shape[0]
    def __getitem__(self, index) -> Tuple[str, str]:
        query_id = self.qrels_df.loc[index]['query_id']
        doc_id = self.qrels_df.loc[index]['doc_id']
        query = self.queries_df[self.queries_df['query_id'] == query_id]['text'].values[0]
        doc = self.docstore.get(doc_id).text
        return query, doc

@dataclass
class DataCollatorForTest(DataCollatorWithPadding):

    max_len: int = 256
    
    def __call__(self, features: List[Tuple[str, str]]) -> Dict[str, Any]:

        query = [f[0] for f in features]
        document = [f[1] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(document[0], list):
            document = sum(document, [])

        batch = self.tokenizer(
                query,
                document,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )

        return batch