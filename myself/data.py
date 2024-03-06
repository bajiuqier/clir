import math
import random
import datasets
import pathlib
from dataclasses import dataclass
from typing import List, Tuple
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding





class TrainDataset(Dataset):
    def __init__(self, data, passage_size=2):
        self.dataset = datasets.load_dataset('json', data_files=data, split='train')
        self.passage_size = passage_size
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]['query']

        passages = []
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)
    
        if len(self.dataset[item]['neg']) < self.passage_size - 1:
            num = math.ceil((self.passage_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.passage_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.passage_size - 1)
        
        passages.extend(negs)

        return query, passages
    
@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    # features 是一个元组。
    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]

        # 将query和passage列表扁平化,处理可能存在的嵌套列表情况。
        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated}


