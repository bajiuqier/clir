from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import DataCollatorWithPadding, AutoTokenizer
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from argments import parse_args


class CLIRMatrixDataset(Dataset):
    def __init__(self, args: parse_args) -> None:
        super().__init__()

        self.dataset = load_dataset('json', data_files=args.train_file, split='train')

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> Tuple[str, list[str]]:
        item = self.dataset[index]
        query = item['query']
        documet = []
        documet.append(item['pos'][0][0])
        documet.append(item['neg'][0][0])

        return query, documet
    

@dataclass
class CLIRMatrixCollator(DataCollatorWithPadding):

    query_max_len: int = 32
    document_max_len: int = 128
    
    def __call__(self, features: List[Tuple[str, list[str]]]) -> Dict[str, Any]:

        query = [f[0] for f in features]
        document = [f[1] for f in features]
        
        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(document[0], list):
            document = sum(document, [])
        
        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            document,
            padding=True,
            truncation=True,
            max_length=self.document_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated}



