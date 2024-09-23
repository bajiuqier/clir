import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from transformers import BertTokenizer, BertModel
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, NamedTuple
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch

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
    def __init__(self, tokenizer, model_args: add_model_args, batch_size: int=8):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.batch_size = batch_size
        self.encoder = BertModel.from_pretrained(model_args.model_name_or_path)
        self.tokenizer = tokenizer
        self.hidden_size = self.encoder.config.hidden_size
        self.normalized = False

        # 定义图神经网络 GNN
        # self.gcn1 = GCNConv(self.hidden_size, 128)
        # self.gcn2 = GCNConv(128, self.hidden_size)
        self.gat1 = GATConv(self.hidden_size, 128, heads=4, dropout=0.1)
        self.gat2 = GATConv(128*4, self.hidden_size, heads=1, concat=True, dropout=0.1)

        # 查询知识融合
        self.query_knowledge_fusion = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.num_entities = 3
        self.classifier = nn.Linear(self.hidden_size * 2, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)
        if self.training:
            # self.loss_function = PairwiseHingeLoss()
            self.loss_function = nn.CrossEntropyLoss()

    def create_subgraph_data(self, V_s, V_t):
        '''
        构建一个子图
        '''
        # 创建节点特征张量
        x = torch.cat([V_s, V_t], dim=0)

        # 边的定义
        # Create edge indices
        num_s = V_s.size(0)
        num_t = V_t.size(0)
        # V_s 的索引是 0 到 num_s - 1
        # V_t 的索引是 num_s 到 num_s + num_t - 1

        # 连接查询对应的源语言实体和目标语言实体
        # edge_index = [[0, num_s]]
        # 或者连接所有的源语言实体和目标语言实体 和上面的是两种图结构
        edge_index = [[i, i + num_s] for i in range(num_s)]

        # 连接查询对应的源语言实体的相邻源语言实体
        for i in range(1, num_s):
            edge_index.append([0, i])        
        # 连接查询对应的目标语言实体的相邻目标语言实体
        for i in range(1, num_t):
            edge_index.append([num_s, num_s + i])        
        # Make edges bidirectional
        edge_index = edge_index + [[j, i] for i, j in edge_index]   

        # 将 edge_index 转换为张量
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # 创建图数据对象
        data = Data(x=x, edge_index=edge_index)

        return data

    
    def construct_graph_batch_data(self, V_s_batch, V_t_batch):
        '''
        构建图的批处理数据 加速计算
        '''
        data_list = []

        for i in range(V_s_batch.size(0)):
            data = self.create_subgraph_data(V_s_batch[i], V_t_batch[i])
            data_list.append(data)

        batch_data = Batch.from_data_list(data_list)

        return batch_data
    
    def extract_sentence_embeddings(self, last_hidden_state, input_ids, token_type_ids):
        """
        提取句子1和句子2的嵌入。
        
        Args:
            last_hidden_state (Tensor): 模型的最后一层隐藏状态，形状为 (batch_size, seq_len, hidden_size)。
            input_ids (Tensor): 输入的 token IDs，形状为 (batch_size, seq_len)。
            token_type_ids (Tensor): token_type_ids，形状为 (batch_size, seq_len)。

        Returns:
            sentence1_embeddings (Tensor): 句子1的嵌入，形状为 (batch_size, hidden_size)。
            sentence2_embeddings (Tensor): 句子2的嵌入，形状为 (batch_size, hidden_size)。
        """
        batch_size, seq_len, hidden_size = last_hidden_state.size()
        sentence1_embeddings = []
        sentence2_embeddings = []
        
        for i in range(batch_size):
            # 获取每个样本的 input_ids 和 token_type_ids
            input_ids_i = input_ids[i]
            token_type_ids_i = token_type_ids[i]
            hidden_states_i = last_hidden_state[i]
            
            # 找到所有 [SEP] token 的位置
            sep_token_id = self.tokenizer.sep_token_id
            sep_positions = (input_ids_i == sep_token_id).nonzero(as_tuple=False).squeeze()
            
            if sep_positions.ndim == 0:
                sep_positions = sep_positions.unsqueeze(0)
            
            if sep_positions.numel() < 2:
                raise ValueError(f"样本 {i} 缺少足够的 [SEP] 标记。")
            
            first_sep = sep_positions[0].item()
            second_sep = sep_positions[1].item()
            
            # 提取句子1的嵌入（[CLS] 后到第一个 [SEP] 之间）
            sentence1_hidden = hidden_states_i[1:first_sep]  # shape: (len1, hidden_size)
            # 提取句子2的嵌入（第一个 [SEP] 后到第二个 [SEP] 之间）
            sentence2_hidden = hidden_states_i[first_sep + 1:second_sep]  # shape: (len2, hidden_size)
            
            # 平均池化
            sentence1_pooled = sentence1_hidden.mean(dim=0)  # shape: (hidden_size)
            sentence2_pooled = sentence2_hidden.mean(dim=0)  # shape: (hidden_size)
            
            sentence1_embeddings.append(sentence1_pooled)
            sentence2_embeddings.append(sentence2_pooled)
        
        # 将列表转换为张量
        sentence1_embeddings = torch.stack(sentence1_embeddings)  # shape: (batch_size, hidden_size)
        sentence2_embeddings = torch.stack(sentence2_embeddings)  # shape: (batch_size, hidden_size)
        
        return sentence1_embeddings, sentence2_embeddings


    def compute_similarity(self, q_representation, d_representation):
        if self.normalized:
            if len(d_representation.size()) == 2:
                return torch.matmul(q_representation, d_representation.transpose(0, 1))
            return torch.matmul(q_representation, d_representation.transpose(-2, -1))
        else:
            return torch.cosine_similarity(q_representation.unsqueeze(1),
                                           d_representation.unsqueeze(0),
                                           dim=-1)

    def forward(self, qd_batch, ed_s_batch, ed_t_batch):

        query_doc_output = self.encoder(**qd_batch)
        # 提取 query 和 doc 的嵌入
        # shape [2 * self.batch_size, self.hidden_size] -> [16, 768]
        V_q, V_d = self.extract_sentence_embeddings(
            query_doc_output.last_hidden_state,
            qd_batch['input_ids'],
            qd_batch['token_type_ids']
        )

        entity_desc_s_output = self.encoder(**ed_s_batch)
        entity_desc_t_output = self.encoder(**ed_t_batch)

        # [(1 + self.num_entities) * self.batch_size, self.hidden_size] -> [32, 768]
        V_entity_desc_s = entity_desc_s_output.last_hidden_state[:, 0, :]
        # [(1 + self.num_entities) * self.batch_size, self.hidden_size] -> [32, 768]
        V_entity_desc_t = entity_desc_t_output.last_hidden_state[:, 0, :]


        if self.training:
            # [16, 4, 768]
            V_entity_desc_s_dim_trans = V_entity_desc_s.view(self.batch_size, (1 + self.num_entities), self.hidden_size).repeat_interleave(2, dim=0)
            # [16, 4, 768]
            V_entity_desc_t_dim_trans = V_entity_desc_t.view(self.batch_size, (1 + self.num_entities), self.hidden_size).repeat_interleave(2, dim=0)
        else:
            # [8, 4, 768]
            V_entity_desc_s_dim_trans = V_entity_desc_s.view(self.batch_size, (1 + self.num_entities), self.hidden_size)
            # [8, 4, 768]
            V_entity_desc_t_dim_trans = V_entity_desc_t.view(self.batch_size, (1 + self.num_entities), self.hidden_size)

        # 获取 图 结构数据
        # graph_batch_data: DataBatch(x=[128, 768], edge_index=[2, 224], batch=[128], ptr=[17])
        graph_batch_data = self.construct_graph_batch_data(V_entity_desc_s_dim_trans, V_entity_desc_t_dim_trans)
        '''
        ### 1. `x=[128, 768]`
        - **`x` 是节点特征矩阵**，其中每个节点都有一个特征向量。
        - `128` 表示 **128个节点**（从所有图中合并）。
        - `768` 表示 **每个节点的特征维度是 768**。这通常是节点嵌入的维度，比如如果使用 BERT，嵌入维度通常是 768。

        ### 2. `edge_index=[2, 224]`
        - **`edge_index` 是边的索引矩阵**，用于定义图中的边。
        - 它是一个大小为 `[2, num_edges]` 的矩阵，**每一列表示一条边**，由两个整数（起始节点索引，终止节点索引）组成。
        - `224` 表示这个批次中的 **224 条边**，这些边连接了上面提到的 `128` 个节点。
        - 第一行表示每条边的源节点。
        - 第二行表示每条边的目标节点。

        ### 3. `batch=[128]`
        - **`batch` 是批次向量**，用来标记每个节点属于哪个图。
        - `128` 表示共有 `128` 个节点，它的长度也是 128。
        - `batch[i]` 的值是节点 `i` 属于的图的编号。比如如果 `batch[0] = 0`，那么节点 0 属于第一个图；`batch[127] = 16`，说明节点 127 属于第 16 个图。

        ### 4. `ptr=[17]`
        - **`ptr` 是指针数组**，表示批次中的每个图是如何划分的。
        - 它的长度为 `num_graphs + 1`。例如，`ptr=[0, 17]` 表示第一个图的节点范围是从 `0` 到 `16`，第二个图的节点范围从 `17` 开始。
        - 在你给出的 `ptr=[17]` 中，`17` 通常是用来表明第一个图的节点数量，后面可能还有更多图的数据没有被展示。
        '''
        graph_batch_data.to(self.device)
        x, edge_index = graph_batch_data.x, graph_batch_data.edge_index
        x = self.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)

        # 使用全局平均池化获取整个图的向量表示
        V_kg = global_mean_pool(x, graph_batch_data.batch)

        # 使用全局最大池化获取整个图的向量表示
        # V_kg = global_max_pool(x, graph_batch_data.batch)

        # 使用全局加和池化获取整个图的向量表示
        # V_kg = global_add_pool(x, graph_batch_data.batch)


        V_q_kg = self.query_knowledge_fusion(torch.cat((V_q, V_kg), dim=-1))
        V_q_kg = self.tanh(V_q_kg)
        V_q_kg = self.dropout(V_q_kg)

        scores = self.compute_similarity(V_q_kg, V_d)
        '''
        从这里卡掉了！！！！
        '''

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


if __name__ == "__main__":

    from data import DatasetForMe, DataCollatorForMe
    from argments import add_logging_args, add_training_args


    logging_args = add_logging_args()
    model_args = add_model_args()
    training_args = add_training_args()

    tokenizer = BertTokenizer.from_pretrained(
        model_args.model_name_or_path,
        clean_up_tokenization_spaces=True,
        use_fast=not model_args.use_slow_tokenizer,
        trust_remote_code=model_args.trust_remote_code
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = MyModel(tokenizer=tokenizer, model_args=model_args, batch_size=training_args.batch_size).to(device)


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


