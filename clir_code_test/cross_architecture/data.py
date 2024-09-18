from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import DataCollatorWithPadding
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
        documet.append(item['pos'][0])
        documet.append(item['neg'][0])

        return query, documet
    

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

