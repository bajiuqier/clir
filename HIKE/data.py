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

        item = self.dataset[query_index]
        entity_s = []
        entity_t = []
        desc_s = []
        desc_t = []

        entity_s.append(item['q_item_info']['label_zh'])
        for em in item['adj_item_info']['label_zh']:
            entity_s.append(em)
        entity_t.append(item['q_item_info']['label_kk'])
        for em in item['adj_item_info']['label_kk']:
            entity_t.append(em)

        desc_s.append(item['q_item_info']['description_zh'])
        for em in item['adj_item_info']['description_zh']:
            desc_s.append(em)
        desc_t.append(item['q_item_info']['description_kk'])
        for em in item['adj_item_info']['description_kk']:
            desc_t.append(em)

        return [query, documet], [entity_s, desc_s], [entity_t, desc_t] 

# test_dataset = DatasetForTest(dataset_file=dataset_file, test_qrels_file=test_qrels_file)
# print(test_dataset[0])

class MyDataset(Dataset):
    def __init__(self, dataset_file):
        super().__init__()
        self.dataset = load_dataset('json', data_files=dataset_file)['train']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[list[str, list[str]], list[list[str], list[str]], list[list[str], list[str]]]:
        item = self.dataset[idx]
        query = item['query']
        documet = []
        documet.append(item['pos_doc'][0])
        documet.append(item['neg_doc'][0])
        entity_s = []
        entity_t = []
        desc_s = []
        desc_t = []

        entity_s.append(item['q_item_info']['label_zh'])
        for em in item['adj_item_info']['label_zh']:
            entity_s.append(em)
        entity_t.append(item['q_item_info']['label_kk'])
        for em in item['adj_item_info']['label_kk']:
            entity_t.append(em)

        desc_s.append(item['q_item_info']['description_zh'])
        for em in item['adj_item_info']['description_zh']:
            desc_s.append(em)
        desc_t.append(item['q_item_info']['description_kk'])
        for em in item['adj_item_info']['description_kk']:
            desc_t.append(em)

        return [query, documet], [entity_s, desc_s], [entity_t, desc_t] 

@dataclass
class DataCollatorForHIKE(DataCollatorWithPadding):

    max_len: int = 256
    training: bool = True
    
    def __call__(self, features: List[Tuple[list[str, list[str]], list[list[str], list[str]], list[list[str], list[str]]]]) -> Dict[str, Any]:

        queries = [f[0][0] for f in features]
        documents = [f[0][1] for f in features]

        entities_s = [f[1][0] for f in features]
        descs_s = [f[1][1] for f in features]

        entities_t = [f[2][0] for f in features]
        descs_t = [f[2][1] for f in features]

        if isinstance(queries[0], list):
            queries = sum(queries, [])
        if isinstance(documents[0], list):
            documents = sum(documents, [])

        if isinstance(entities_s[0], list):
            entities_s = sum(entities_s, [])
        if isinstance(descs_s[0], list):
            descs_s = sum(descs_s, [])

        if isinstance(entities_t[0], list):
            entities_t = sum(entities_t, [])
        if isinstance(descs_t[0], list):
            descs_t = sum(descs_t, [])
        
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
        ed_s_batch = self.tokenizer(
                entities_s,
                descs_s,
                padding=True,
                truncation='only_second',
                max_length=64,
                # max_length=self.max_len,
                return_tensors="pt",
            )
        ed_t_batch = self.tokenizer(
                entities_t,
                descs_t,
                padding=True,
                truncation='only_second',
                max_length=64,
                # max_length=self.max_len,
                return_tensors="pt",
            )

        return {'qd_batch': qd_batch, 'ed_s_batch': ed_s_batch, 'ed_t_batch': ed_t_batch}







# dataset = MyDataset(dataset_file=dataset_file)

# model_path = str(Path(__file__).parent.parent / 'models' / 'models--bert-base-multilingual-uncased')
# encoder = BertModel.from_pretrained(model_path)
# tokenizer = BertTokenizer.from_pretrained(model_path, use_slow_tokenizer=False)

# data_collator = DataCollatorForMe(tokenizer, max_len=256)

# train_dataloader = DataLoader(
#     dataset, shuffle=True, collate_fn=data_collator, batch_size=8
# )

# print(train_dataloader)