#!/usr/bin/env python
# coding: utf-8


import time, datetime, os, json, glob
import pandas as pd
import numpy as np
from collections import defaultdict
from abc import ABC
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, PreTrainedModel,                     AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, Dataset, DataLoader, RandomSampler
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer
from transformers.integrations import TensorBoardCallback, TrainerCallback
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig
from functools import partial
from typing import List, Dict, Tuple
from pathlib import Path
import random
import logging
logging.disable(logging.INFO)
logging.disable(logging.WARNING)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# ------------------------ parameters ------------------------
CHPT_INTERVAL = 10000

train_data_file = './ProcessedData/final_prf1_clirmatrix_de_bi139-full_en_train.csv'
num_feedback_docs = 1
model_type = "microsoft/xlm-align-base"
save_dir = './Trained_prf1_xlmalign'
lang = 'ru'
results_dir_path = save_dir + f"_{lang}/"

train_limit = 160000
sim_func = 'dot'
cache_dir = './cache_dir/'
learning_rate = 5e-6
weight_decay = 0.1
num_epochs = 4
batch_size = 16
disable_tqdm = False
save_steps = 80000
save_total_limit = 5
fp16 = True
logging_steps = 1
max_seq_length = 180
num_neg_samples = 1
# -------------------------------------------------------------


# 1. -------------------------- load data ---------------------
train_df = pd.read_csv(train_data_file)
print("Number of training samples : ", train_df.shape[0])
train_df = train_df.iloc[:train_limit]
print("Number of actual training samples after limitation : ", train_df.shape[0])


class MyDataset(Dataset):
    def __init__(self, data, num_feedback_docs=2):
        self.num_feedback_docs = num_feedback_docs
        self.data_df = data        
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> [str, str]:
        index = idx % self.__len__()
        return_list = [self.data_df.loc[index]['query_text'], 
                       self.data_df.loc[index]['doc_text']] + [self.data_df.loc[index][f'feedback_doc{i+1}_text'] for i in range(self.num_feedback_docs)]
        return return_list
    
train_dataset = MyDataset(train_df, num_feedback_docs=num_feedback_docs)


@dataclass
class MyCollator(DataCollatorWithPadding):
    max_seq_length: int = 180
    num_neg_samples: int = 1
    num_feedback_docs: int = 2

    def __post_init__(self):
        pass      
                
    def __call__(self, examples: List[Tuple[str, str]]) -> None:  

        # 随机生成负样本      
        derange_list = []
        for _ in range(self.num_neg_samples):
            derange = list(range(len(examples)))        
            while any([ i==e for i, e in enumerate(derange) ]):
                random.shuffle(derange)
            derange_list.append(derange)
            
        ### documents generation
        docs = []
        for exmp_ids, example in enumerate(examples):
            docs.append( [example[1]] + [examples[derange_list[neg_idx][exmp_ids]][1] for neg_idx in range(self.num_neg_samples)] )
        docs = sum(docs, [])

        positive_idx_per_query = [i * (self.num_neg_samples+1) for i in range(len(examples))]
                    
        ### queries generation
        queries = []
        for i in range(self.num_feedback_docs):
            curr_pairs = [[example[0], example[i+2]] for example in examples]
            queries.extend(curr_pairs)
            
        docs_encoded = self.tokenizer.batch_encode_plus(
            docs,
            truncation="longest_first", 
            max_length = self.max_seq_length,
            padding="max_length", 
            add_special_tokens=True, 
            return_tensors="pt", 
            return_attention_mask=True, 
            return_token_type_ids=False,
            return_special_tokens_mask=False
            )
        
        queries_encoded = self.tokenizer.batch_encode_plus(
            queries,
            truncation="longest_first", 
            max_length = self.max_seq_length,
            padding="max_length", 
            add_special_tokens=True, 
            return_tensors="pt", 
            return_attention_mask=True, 
            return_token_type_ids=False,
            return_special_tokens_mask=False
            )

        batch = {
            "docs_input_ids": docs_encoded['input_ids'],
            "docs_attention_mask": docs_encoded['attention_mask'],
            "queries_input_ids": queries_encoded['input_ids'],
            "queries_attention_mask": queries_encoded['attention_mask'],
            "positive_idx_per_query": torch.tensor(positive_idx_per_query),
            "es": len(examples)
        }
        return batch
    
    
tokenizer = AutoTokenizer.from_pretrained(model_type)
data_collator = MyCollator(tokenizer=tokenizer,
                               max_seq_length=max_seq_length,
                               num_neg_samples=num_neg_samples,
                               num_feedback_docs=num_feedback_docs)


def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r

def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)
    

class BiEncoderNllLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        sim_func: str = 'dot',
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        scores = self.get_scores(q_vectors, ctx_vectors, sim_func)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        if loss_scale:
            loss.mul_(loss_scale)

        return loss

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T, sim_func: str) -> T:
        if sim_func == 'cosine':
            f = BiEncoderNllLoss.get_cos_similarity_function()
            return f(q_vector, ctx_vectors)
        else:
            f = BiEncoderNllLoss.get_dot_similarity_function()
            return f(q_vector, ctx_vectors)

    @staticmethod
    def get_dot_similarity_function():
        return dot_product_scores
    @staticmethod
    def get_cos_similarity_function():
        return cosine_scores


class DualEncoder(nn.Module):    
    def __init__(
        self,
        lm_model,
        num_feedback_docs=1,
        sim_func='dot',
        is_train=True,
        cache_dir=None
    ):
        super(DualEncoder, self).__init__()
        self.lm_model = lm_model
        self.num_feedback_docs = num_feedback_docs
        
        self.is_train = is_train
        if self.is_train:
            self.sim_func = sim_func
            self.loss_function = BiEncoderNllLoss()

    def forward(self, queries_input_ids, queries_attention_mask,
                docs_input_ids, docs_attention_mask, positive_idx_per_query, es):
        query_out = self.lm_model(
            input_ids=queries_input_ids,
            attention_mask=queries_attention_mask,
            output_hidden_states=False,
            return_dict=True
            )
        doc_out = self.lm_model(
            input_ids=docs_input_ids,
            attention_mask=docs_attention_mask,
            output_hidden_states=False,
            return_dict=True
            )
        query_cls_emb = query_out['last_hidden_state'][:, 0, :]
        doc_cls_emb = doc_out['last_hidden_state'][:, 0, :]
        
        if not self.is_train:
            output_dict = {'query_vector': query_cls_emb, 'doc_vector': doc_cls_emb}
            return output_dict
            
        else:
            concat_list = []
            for i in range(self.num_feedback_docs):
                concat_list.append( query_cls_emb[ es*(i): es*(i+1)] )
                                
            query_cls_emb = torch.mean( torch.stack(concat_list, dim=0) , dim=0)
                                
            loss = self.loss_function.calc(query_cls_emb, doc_cls_emb, positive_idx_per_query, sim_func=self.sim_func)
            output_dict = {'loss': loss, 'query_vector': query_cls_emb, 'doc_vector': doc_cls_emb}
            return output_dict
        
    
    def save_pretrained(self, output_dir: str):
        self.lm_model.save_pretrained(os.path.join(output_dir, "lm_model"))
        
    @classmethod
    def from_pretrained(cls, saved_dir, sim_func='dot', is_train=False, cache_dir=None):
        lm_model = AutoModel.from_pretrained(os.path.join(saved_dir, "lm_model"))
        model = cls(lm_model, sim_func, is_train, cache_dir)
        return model
    
    
# 2. -------------------------- load tokenizer & pre-trained model ---------------------
lm_model = AutoModel.from_pretrained(model_type, cache_dir=cache_dir)
model = DualEncoder(lm_model=lm_model, sim_func=sim_func, num_feedback_docs=num_feedback_docs, is_train=True, cache_dir=cache_dir)
model.to(device)
print("Model loaded into GPU")


class MyCallback(TrainerCallback):
    def on_step_end(self, args, state, control, logs=None, model=None, **kwargs):
        opath = Path(args.output_dir)
        save_path = opath / f"savechpt-{state.global_step}"
        if state.global_step % CHPT_INTERVAL == 0:
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            print(f"Checkpoint for {state.global_step} saved")


training_args = TrainingArguments(
        output_dir=results_dir_path,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        disable_tqdm=disable_tqdm,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        fp16=fp16,
        logging_steps=logging_steps)


# 3. -------------------------- start training ---------------------
trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[TensorBoardCallback, MyCallback]
        )

trainer.train()

