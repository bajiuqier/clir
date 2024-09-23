import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from transformers import BertTokenizer, BertModel
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, NamedTuple
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
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

class MyModel(nn.Module):
    def __init__(self, model_args: add_model_args):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.batch_size = 8
        self.encoder = BertModel.from_pretrained(model_args.model_name_or_path)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_heads = 8
        # self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.num_heads, dropout=0.1, batch_first=True)

        # self.gcn1 = GCNConv(self.hidden_size, 128)
        # self.gcn2 = GCNConv(128, self.hidden_size)
        self.gat1 = GATConv(self.hidden_size, 128, heads=4, dropout=0.1)
        self.gat2 = GATConv(128*4, self.hidden_size, heads=1, concat=True, dropout=0.1)

        self.num_entities = 3
        # self.knowledge_level_fusion = nn.Linear(self.hidden_size * (2 + self.num_entities), self.hidden_size)
        # self.language_level_fusion = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size * 2, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)
        if self.training:
            self.loss_function = PairwiseHingeLoss()
    def create_subgraph_data(self, V_qd, V_s, V_t):
        # 创建节点特征张量
        x = torch.cat([V_qd.unsqueeze(0), V_s, V_t], dim=0)
        
        # 边的定义
        # Create edge indices
        num_s = V_s.size(0)
        num_t = V_t.size(0)

        # V_qd 的索引是0，V_s 的索引是 1 到 num_s，V_t 的索引是 num_s + 1 到 num_s + num_t
        # Connect v_qd to its children
        edge_index = [[0, 1], [0, num_s + 1]]
        
        # Connect v_s nodes
        for i in range(1, num_s):
            edge_index.append([1, i + 1])
        
        # Connect v_t nodes
        for i in range(1, num_t):
            edge_index.append([num_s + 1, num_s + 1 + i])
        
        # Make edges bidirectional
        edge_index = edge_index + [[j, i] for i, j in edge_index]
        
        # 将 edge_index 转换为张量
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # 创建图数据对象
        data = Data(x=x, edge_index=edge_index)

        return data

    # 构建批处理数据
    def construct_batch(self, V_qd_batch, V_s_batch, V_t_batch):
        data_list = []
        for i in range(V_qd_batch.size(0)):
            data = self.create_subgraph_data(V_qd_batch[i], V_s_batch[i], V_t_batch[i])
            data_list.append(data)
        batch_data = Batch.from_data_list(data_list)
        return batch_data

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

        # 获取 图 结构数据
        batch_data = self.construct_batch(query_doc_embedding, entity_desc_s_embedding_dim_trans, entity_desc_t_embedding_dim_trans)
        batch_data.to(self.device)
        x, edge_index = batch_data.x, batch_data.edge_index
        x = F.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)

        V_qd_list = []
        for i in range(batch_data.batch_size):
            batch_mask = batch_data.batch == i
            V_qd_list.append(x[batch_mask][0])

        v_qd = torch.stack(V_qd_list)

        # # [16, 5, 768]
        # knowledge_input_s = torch.cat((query_doc_embedding_dim_trans, entity_desc_s_embedding_dim_trans), dim=1)
        # knowledge_input_s = torch.cat((query_doc_embedding_dim_trans, entity_desc_t_embedding_dim_trans), dim=1)

        # # Knowledge-level fusion
        # # [16, 5, 768]
        # attn_output_s, _ = self.multihead_attn(knowledge_input_s, knowledge_input_s, knowledge_input_s)
        # attn_output_s = self.dropout(attn_output_s)
        # attn_output_t, _ = self.multihead_attn(knowledge_input_s, knowledge_input_s, knowledge_input_s)
        # attn_output_t = self.dropout(attn_output_t)


        # if self.training:
        #     # self.knowledge_level_fusion 的输入维度是 [16, (2+3)*768]
        #     # knowledge_fused_ -> [16, 768]
        #     knowledge_fused_s = self.tanh(self.knowledge_level_fusion(attn_output_s.reshape(2 * self.batch_size, (2 + self.num_entities) * self.hidden_size)))
        #     knowledge_fused_t = self.tanh(self.knowledge_level_fusion(attn_output_s.reshape(2 * self.batch_size, (2 + self.num_entities) * self.hidden_size)))
        # else:
        #     knowledge_fused_s = self.tanh(self.knowledge_level_fusion(attn_output_s.reshape(self.batch_size, (2 + self.num_entities) * self.hidden_size)))
        #     knowledge_fused_t = self.tanh(self.knowledge_level_fusion(attn_output_s.reshape(self.batch_size, (2 + self.num_entities) * self.hidden_size)))

        # knowledge_fused_s = self.dropout(knowledge_fused_s)
        # knowledge_fused_t = self.dropout(knowledge_fused_t)

        # # Language-level fusion
        # # self.language_level_fusion 的输入维度是 [16, 3*768]
        # # language_fused -> [16, 768]
        # language_fused = self.tanh(self.language_level_fusion(torch.cat((query_doc_embedding, knowledge_fused_s, knowledge_fused_t), dim=-1)))
        # language_fused = self.dropout(language_fused)
        # Classificatiot
        # self.classifier 的输入维度是 [16, 2*768]
        # scores -> [16]
        # scores = self.classifier(torch.cat((query_doc_embedding, language_fused), dim=-1))
        scores = self.classifier(torch.cat((query_doc_embedding, v_qd), dim=-1))

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

from data import DatasetForMe, DataCollatorForMe
from argments import add_logging_args, add_training_args


logging_args = add_logging_args()
model_args = add_model_args()
training_args = add_training_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = MyModel(model_args=model_args).to(device)

tokenizer = BertTokenizer.from_pretrained(
    model_args.model_name_or_path, use_fast=not model_args.use_slow_tokenizer, trust_remote_code=model_args.trust_remote_code
)

print("  train_dataset生成中ing......")
train_dataset = DatasetForMe(dataset_file=training_args.train_dataset_name_or_path, dataset_type='train')
test_dataset = DatasetForMe(dataset_file=training_args.test_dataset_name_or_path, dataset_type='test', test_qrels_file=training_args.test_qrels_file)

train_data_collator = DataCollatorForMe(tokenizer, max_len=256, training=True)
test_data_collator = DataCollatorForMe(tokenizer, max_len=256, training=False)

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=train_data_collator, batch_size=training_args.batch_size, drop_last=True
)
test_dataloader = DataLoader(
    test_dataset, shuffle=False, collate_fn=test_data_collator, batch_size=training_args.batch_size, drop_last=True
)

for batch_idx, batch in enumerate(train_dataloader):
    qd_batch = {k: v.to(device) for k, v in batch['qd_batch'].items()}
    ed_s_batch = {k: v.to(device) for k, v in batch['ed_s_batch'].items()}
    ed_t_batch = {k: v.to(device) for k, v in batch['ed_t_batch'].items()}

    outputs = model(
        qd_batch=qd_batch,
        ed_s_batch=ed_s_batch,
        ed_t_batch=ed_t_batch
    )

    print(outputs)


