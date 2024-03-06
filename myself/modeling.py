from dataclasses import dataclass
from typing import Dict, Optional
from torch import nn, Tensor
import torch
from transformers import AutoModel
from transformers.file_utils import ModelOutput


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class DualEncoder(nn.Module):    
    def __init__(
        self,
        lm_model: str = None,
        is_train: bool = True,
        sentence_pooling_method: str = 'cls',

    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(lm_model)
        self.lm_model = lm_model
        
        self.is_train = is_train
        if self.is_train:
            # self.sim_func = sim_func
            self.loss_function = nn.CrossEntropyLoss(reduction='mean')
            
    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        
        
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