from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from transformers import DataCollatorWithPadding
from dataclasses import dataclass
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import ir_datasets
# from argments import add_training_args


class DatasetForTest(Dataset):
    def __init__(self, dataset_file, test_qrels_file):
        super().__init__()
        self.dataset = load_dataset('json', data_files=dataset_file)['train']
        self.dataset_query_id = self.dataset[:]['query_id']
        self.test_qrels = pd.read_csv(test_qrels_file, encoding='utf-8')
        self.test_qrels['query_id'] = self.test_qrels['query_id'].astype(str)
        self.test_qrels['doc_id'] = self.test_qrels['doc_id'].astype(str)

        self.clir_dataset = ir_datasets.load('clirmatrix/kk/bi139-base/zh/train')
        # self.queries_df = pd.DataFrame(self.clir_dataset.queries_iter())
        self.docstore = self.clir_dataset.docs_store()

    def __len__(self):
        return len(self.test_qrels)

    def __getitem__(self, idx):
        query_id = self.test_qrels.loc[idx]['query_id']
        doc_id = self.test_qrels.loc[idx]['doc_id']

        query_index = self.dataset_query_id.index(query_id)
        # if self.dataset[query_index]['query'] != self.queries_df[self.queries_df['query_id'] == query_id]['text'].values[0]:
        #     raise ValueError("query文本不对应")

        query = self.dataset[query_index]['query']
        documet = self.docstore.get(doc_id).text

        return query, documet

class DatasetForMBERT(Dataset):
    def __init__(self, dataset_file, dataset_type: str='train', test_qrels_file: str=None):
        super().__init__()
        self.dataset_type = dataset_type
        self.dataset = load_dataset('json', data_files=dataset_file)['train']
        
        if self.dataset_type == 'test':

            self.dataset_query_id = self.dataset[:]['query_id']
            self.test_qrels = pd.read_csv(test_qrels_file, encoding='utf-8')
            self.test_qrels['query_id'] = self.test_qrels['query_id'].astype(str)
            self.test_qrels['doc_id'] = self.test_qrels['doc_id'].astype(str)

            # 检查 test_dataset.json 和 test_qrels 中的数据是否对等
            if len(self.dataset_query_id) != len(set(self.dataset_query_id)):
                raise ValueError("test_dataset.json 中的 query_id 存在 重复数据")

            if len(set(self.dataset_query_id)) != len(set(self.test_qrels['query_id'])):
                raise ValueError("test_dataset.json 中的 query_id 数据 与 test_qrels 中的 query_id 数据 不对等")

            self.clir_dataset = ir_datasets.load('clirmatrix/kk/bi139-base/zh/test1')
            # self.queries_df = pd.DataFrame(self.clir_dataset.queries_iter())
            self.docstore = self.clir_dataset.docs_store()

    def __len__(self):
        if self.dataset_type == 'train':
            return len(self.dataset)
        if self.dataset_type == 'test':
            return len(self.test_qrels)

    def __getitem__(self, idx) -> Union[Tuple[str, list[str]], Tuple[str, str]]:

        if self.dataset_type == 'train':
            item = self.dataset[idx]
            query = item['query']
            documet = []
            documet.append(item['pos_doc'][0])
            documet.append(item['neg_doc'][0])

        if self.dataset_type == 'test':
            query_id = self.test_qrels.loc[idx]['query_id']
            doc_id = self.test_qrels.loc[idx]['doc_id']

            # 获取 query_id 在 test_dataset.json 中的 位置
            query_index = self.dataset_query_id.index(query_id)

            query = self.dataset[query_index]['query']
            documet = self.docstore.get(doc_id).text

        return query, documet

@dataclass
class DataCollatorForMBERT(DataCollatorWithPadding):

    max_len: int = 256
    training: bool = True
    
    def __call__(self, features: List[Tuple[str, list[str]]]) -> Dict[str, Any]:

        queries = [f[0] for f in features]
        documents = [f[1] for f in features]

        if isinstance(queries[0], list):
            queries = sum(queries, [])
        if isinstance(documents[0], list):
            documents = sum(documents, [])
        
        if self.training:
            # 将 query 中的内容 重复两份
            queries = sum([[element]*2 for element in queries], [])

        qd_batch = self.tokenizer(
                queries,
                documents,
                padding=True,
                truncation='only_second',
                max_length=self.max_len,
                return_tensors="pt",
            )


        return {'qd_batch': qd_batch}





# training_args = add_training_args()
# test_dataset = DatasetForMBERT(dataset_file=training_args.train_dataset_name_or_path, dataset_type='test', test_qrels_file=training_args.test_qrels_file)

# dd = test_dataset[0]

# print(dd)

