from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from dataclasses import dataclass
import pandas as pd
from typing import Dict, List, Tuple, Any, Union
import ir_datasets


class DatasetForMe(Dataset):
    def __init__(self, dataset_file, dataset_type: str = 'train', test_qrels_file: str = None):
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

    def __getitem__(self, idx) -> Tuple[
        Tuple[str, Union[str, List[str]]], Tuple[List[str], List[str]], Tuple[List[str], List[str]]]:

        if self.dataset_type == 'train':
            item = self.dataset[idx]
            query = item['query']
            document = []
            document.append(item['pos_doc'][0])
            document.append(item['neg_doc'][0])

        if self.dataset_type == 'test':
            query_id = self.test_qrels.loc[idx]['query_id']
            doc_id = self.test_qrels.loc[idx]['doc_id']

            query_index = self.dataset_query_id.index(query_id)
            query = self.dataset[query_index]['query']
            document = self.docstore.get(doc_id).text

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

        return (query, document), (entity_s, desc_s), (entity_t, desc_t)


@dataclass
class DataCollatorForMe(DataCollatorWithPadding):
    max_len: int = 256
    training: bool = True

    def __call__(
        self,
        features: List[Tuple[Tuple[str, Union[str, List[str]]], Tuple[List[str], List[str]], Tuple[List[str], List[str]]]]
        ) -> Dict[str, Any]:

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
            queries = sum([[element] * 2 for element in queries], [])

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
            # max_length=64,
            max_length=self.max_len,
            return_tensors="pt",
        )
        ed_t_batch = self.tokenizer(
            entities_t,
            descs_t,
            padding=True,
            truncation='only_second',
            # max_length=64,
            max_length=self.max_len,
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
