from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import BertTokenizer
from transformers import DataCollatorWithPadding
from dataclasses import dataclass

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

dataset_file = str(Path(__file__).parent / 'dataset.jsonl')
# dataset = load_dataset('json', data_files=dataset_file)['train']
# print(dataset)

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
    
# dataset = MyDataset(dataset_file=dataset_file)
# print(dataset)

@dataclass
class DataCollatorForMe(DataCollatorWithPadding):

    max_len: int = 256
    
    def __call__(self, features: List[Tuple[list[str, list[str]], list[list[str], list[str]], list[list[str], list[str]]]]) -> Dict[dict[str, Any]]:

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
        # 将 query 中的内容 重复两份
        queries = sum([[element]*2 for element in queries], [])

        qd_batch = self.tokenizer(
                queries,
                documents,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
        ed_s_batch = self.tokenizer(
                entities_s,
                descs_s,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
        ed_t_batch = self.tokenizer(
                entities_t,
                descs_t,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )

        return {'qd_batch': qd_batch, 'ed_s_batch': ed_s_batch, 'ed_t_batch': ed_t_batch}
