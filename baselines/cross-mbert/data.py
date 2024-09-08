from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from transformers import DataCollatorWithPadding
from dataclasses import dataclass
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import ir_datasets

# dataset_file = str(Path(__file__).parent / 'data' / 'test_dataset.jsonl')
# test_qrels_file = str(Path(__file__).parent / 'data' / 'test_qrels.csv')

# dataset = load_dataset('json', data_files=dataset_file)['train']
# print(dataset)
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

    def __getitem__(self, idx) -> Tuple[list[str, list[str]], list[list[str], list[str]], list[list[str], list[str]]]:
        query_id = self.test_qrels.loc[idx]['query_id']
        doc_id = self.test_qrels.loc[idx]['doc_id']

        query_index = self.dataset_query_id.index(query_id)
        # if self.dataset[query_index]['query'] != self.queries_df[self.queries_df['query_id'] == query_id]['text'].values[0]:
        #     raise ValueError("query文本不对应")

        query = self.dataset[query_index]['query']
        documet = self.docstore.get(doc_id).text

        return query, documet

# test_dataset = DatasetForTest(dataset_file=dataset_file, test_qrels_file=test_qrels_file)
# print(test_dataset[0])

class MyDataset(Dataset):
    def __init__(self, dataset_file):
        super().__init__()
        self.dataset = load_dataset('json', data_files=dataset_file)['train']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[str, list[str]]:
        item = self.dataset[idx]
        query = item['query']
        documet = []
        documet.append(item['pos_doc'][0])
        documet.append(item['neg_doc'][0])


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







# dataset = MyDataset(dataset_file=dataset_file)

# model_path = str(Path(__file__).parent.parent / 'models' / 'models--bert-base-multilingual-uncased')
# encoder = BertModel.from_pretrained(model_path)
# tokenizer = BertTokenizer.from_pretrained(model_path, use_slow_tokenizer=False)

# data_collator = DataCollatorForMe(tokenizer, max_len=256)

# train_dataloader = DataLoader(
#     dataset, shuffle=True, collate_fn=data_collator, batch_size=8
# )

# print(train_dataloader)