import torch
import torch_geometric
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

def construct_graph(V_qd, V_s, V_t):
    # 创建节点特征张量
    x = torch.cat([V_qd.unsqueeze(0), V_s, V_t], dim=0)

    # 边的定义
    # V_qd 的索引是0，V_s 的索引是 1 到 n+1，V_t 的索引是 n+2 到 2n+2
    edge_index = []

    # V_qd 与 V_s 和 V_t 的连接
    for i in range(1, V_s.size(0) + 1):
        edge_index.append([0, i])
        edge_index.append([i, 0])

    for i in range(V_s.size(0) + 1, x.size(0)):
        edge_index.append([0, i])
        edge_index.append([i, 0])

    # 将 edge_index 转换为张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index)
    
    return data


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# 如果使用GAT可以定义如下
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 128, heads=8, dropout=0.6)
        self.conv2 = GATConv(128*8, out_channels, heads=1, concat=True, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def train_model(data, model, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        V_qd_new = out[0]
        # 定义损失函数，这里用余弦相似度和交叉熵损失的组合
        criterion = torch.nn.CrossEntropyLoss()
        cos_sim = F.cosine_similarity(V_qd_new.unsqueeze(0), data.x.unsqueeze(0))
        loss = criterion(cos_sim, torch.tensor([1.0]))  # 目标是最大化相似度
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# 示例数据
V_qd = torch.randn(768)  # 假设768维度
V_s = torch.randn(10, 768)  # 10个中文实体及描述
V_t = torch.randn(10, 768)  # 10个哈萨克语实体及描述

data = construct_graph(V_qd, V_s, V_t)
model = GCN(in_channels=768, out_channels=768)  # 根据需要选择 GCN 或 GAT
# train_model(data, model)
