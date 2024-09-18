import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple, cast


class GetEmbedding:
    def __init__(self, model, tokenizer, batch_size: int=256, max_seq_length: int=512, normalize_embeddings: bool = True, pooling_method: str = 'cls'):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method
        # self.all_embeddings = []

    @torch.no_grad()
    def encode(self, sentences: Union[List[str], str], convert_to_numpy: bool = True):

        self.model.eval()
        # input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            # input_was_string = True

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Inference Embeddings", disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + self.batch_size]

            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.max_seq_length,
            ).to(self.device)

            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])

            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            # 将embeddings转成torch.Tensor数据类型
            embeddings = cast(torch.Tensor, embeddings)

            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.stack(all_embeddings)

        # 下面的操作返回的all_embeddings是一个一维的数据 但是在 do search 时 要求query是一个二维的向量 所以没必要做这一步
        # if input_was_string:
        #     return all_embeddings[0]
        return all_embeddings

    def pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor = None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0, :]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d

    def save_embeddings(self, corpus: Union[List[str], str], embedding_path: str=None):

        corpus_embeddings = self.encode(sentences=corpus, convert_to_numpy=True)
        
        print(f"saving embeddings at {embedding_path}...")
        memmap = np.memmap(
            embedding_path,
            shape=corpus_embeddings.shape,
            mode="w+",
            dtype=corpus_embeddings.dtype
        )

        length = corpus_embeddings.shape[0]
        # add in batch
        save_batch_size = 10000
        if length > save_batch_size:
            # leave=False参数时，这意味着进度条不会保留在终端上，一旦循环结束，进度条就会消失。
            for i in tqdm(range(0, length, save_batch_size), leave=True, desc="Saving Embeddings"):
                j = min(i + save_batch_size, length)
                memmap[i: j] = corpus_embeddings[i: j]
        else:
            memmap[:] = corpus_embeddings

        memmap.flush()
        print('Successfully saved embeddings')