import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, NamedTuple

from pathlib import Path
from argments import add_model_args



class PairwiseHingeLoss(torch.nn.Module):
    def __init__(self, margin=0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, score, target):
        # Compute the loss based on the target labels
        loss = torch.where(target == 1, 1 - score, 
                           torch.clamp(score - self.margin, min=0))
        
        # Apply the reduction method
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

class OutputTuple(NamedTuple):
    loss: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    # query_vector: Optional[torch.Tensor] = None
    # doc_vector: Optional[torch.Tensor] = None
    # embedding: Optional[torch.Tensor] = None

class HIKE(nn.Module):
    def __init__(self, model_args: add_model_args):
        super().__init__()
        self.batch_size = 8
        self.encoder = BertModel.from_pretrained(model_args.model_name_or_path)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_heads = model_args.multi_atten_heads_num
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.num_heads, dropout=0.1, batch_first=True)

        self.num_entities = 3
        self.knowledge_level_fusion = nn.Linear(self.hidden_size * (2 + self.num_entities), self.hidden_size)
        self.language_level_fusion = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size * 2, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)
        if self.training:
            self.loss_function = PairwiseHingeLoss()

    def forward(self, qd_batch, ed_s_batch, ed_t_batch):

        query_doc_output = self.encoder(**qd_batch)
        entity_desc_s_output = self.encoder(**ed_s_batch)
        entity_desc_t_output = self.encoder(**ed_t_batch)

        # [2 * self.batch_size, self.hidden_size] -> [16, 768]
        query_doc_embedding = query_doc_output.last_hidden_state[:, 0, :]
        # [(1 + self.num_entities) * self.batch_size, self.hidden_size] -> [32, 768]
        entity_desc_s_embedding = entity_desc_s_output.last_hidden_state[:, 0, :]
        # [(1 + self.num_entities) * self.batch_size, self.hidden_size] -> [32, 768]
        entity_desc_t_embedding = entity_desc_t_output.last_hidden_state[:, 0, :]

        # 变换 query_doc_embedding entity_desc_s_embedding entity_desc_t_embedding 的维度
        # [16, 1, 768]
        query_doc_embedding_dim_trans = query_doc_embedding.unsqueeze(1)

        if self.training:
            # [16, 4, 768]
            entity_desc_s_embedding_dim_trans = entity_desc_s_embedding.view(self.batch_size, (1 + self.num_entities), self.hidden_size).repeat_interleave(2, dim=0)
            # [16, 4, 768]
            entity_desc_t_embedding_dim_trans = entity_desc_t_embedding.view(self.batch_size, (1 + self.num_entities), self.hidden_size).repeat_interleave(2, dim=0)
        else:
            # [8, 4, 768]
            entity_desc_s_embedding_dim_trans = entity_desc_s_embedding.view(self.batch_size, (1 + self.num_entities), self.hidden_size)
            # [8, 4, 768]
            entity_desc_t_embedding_dim_trans = entity_desc_t_embedding.view(self.batch_size, (1 + self.num_entities), self.hidden_size)

        # [16, 5, 768]
        knowledge_input_s = torch.cat((query_doc_embedding_dim_trans, entity_desc_s_embedding_dim_trans), dim=1)
        knowledge_input_s = torch.cat((query_doc_embedding_dim_trans, entity_desc_t_embedding_dim_trans), dim=1)

        # Knowledge-level fusion
        # [16, 5, 768]
        attn_output_s, _ = self.multihead_attn(knowledge_input_s, knowledge_input_s, knowledge_input_s)
        attn_output_s = self.dropout(attn_output_s)
        attn_output_t, _ = self.multihead_attn(knowledge_input_s, knowledge_input_s, knowledge_input_s)
        attn_output_t = self.dropout(attn_output_t)


        if self.training:
            # self.knowledge_level_fusion 的输入维度是 [16, (2+3)*768]
            # knowledge_fused_ -> [16, 768]
            knowledge_fused_s = self.tanh(self.knowledge_level_fusion(attn_output_s.reshape(2 * self.batch_size, (2 + self.num_entities) * self.hidden_size)))
            knowledge_fused_t = self.tanh(self.knowledge_level_fusion(attn_output_s.reshape(2 * self.batch_size, (2 + self.num_entities) * self.hidden_size)))
        else:
            knowledge_fused_s = self.tanh(self.knowledge_level_fusion(attn_output_s.reshape(self.batch_size, (2 + self.num_entities) * self.hidden_size)))
            knowledge_fused_t = self.tanh(self.knowledge_level_fusion(attn_output_s.reshape(self.batch_size, (2 + self.num_entities) * self.hidden_size)))

        knowledge_fused_s = self.dropout(knowledge_fused_s)
        knowledge_fused_t = self.dropout(knowledge_fused_t)

        # Language-level fusion
        # self.language_level_fusion 的输入维度是 [16, 3*768]
        # language_fused -> [16, 768]
        language_fused = self.tanh(self.language_level_fusion(torch.cat((query_doc_embedding, knowledge_fused_s, knowledge_fused_t), dim=-1)))
        language_fused = self.dropout(language_fused)
        # Classificatiot
        # self.classifier 的输入维度是 [16, 2*768]
        # scores -> [16]
        scores = self.classifier(torch.cat((query_doc_embedding, language_fused), dim=-1))
        if self.training:
            # scores -> [8, 2]
            scores = scores.view(-1, 2)
            # scores -> [8, 2]
            scores = self.softmax(scores)

            target = torch.tensor([1, 0], device=scores.device, dtype=torch.long).repeat(scores.size()[0], 1)
            loss = self.loss_function(scores, target)
        else:
            loss = None

        return OutputTuple(
            loss=loss,
            scores=scores
        )


# batch_size = 8
# dataset_file = str(Path(__file__).parent / 'data' / 'dataset.jsonl')
# model_path = str(Path(__file__).parent.parent / 'models' / 'models--bert-base-multilingual-uncased')

# encoder = BertModel.from_pretrained(model_path)
# model = HIKE(model_name_or_path=model_path)
# tokenizer = BertTokenizer.from_pretrained(model_path)

# dataset = MyDataset(dataset_file=dataset_file)
# data_collator = DataCollatorForMe(tokenizer, max_len=256)

# train_dataloader = DataLoader(
#     dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
# )

# for batch_idx, batch in enumerate(train_dataloader):
#     scores = model(qd_batch=batch['qd_batch'], ed_s_batch=batch['ed_s_batch'], ed_t_batch=batch['ed_t_batch']).scores
#     print(scores)
#     break