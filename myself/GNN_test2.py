import torch
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# 构建每个图的数据
def construct_graph(V_qd, V_s, V_t):
    # 拼接所有节点特征
    x = torch.cat([V_qd.unsqueeze(0), V_s, V_t], dim=0)
    edge_index = []

    # V_qd 与 V_s 和 V_t 的连接
    for i in range(1, V_s.size(0) + 1):
        edge_index.append([0, i])
        edge_index.append([i, 0])

    for i in range(V_s.size(0) + 1, x.size(0)):
        edge_index.append([0, i])
        edge_index.append([i, 0])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    return data

# 构建批处理数据
def construct_batch(V_qd_batch, V_s_batch, V_t_batch):
    data_list = []
    for i in range(V_qd_batch.size(0)):
        data = construct_graph(V_qd_batch[i], V_s_batch[i], V_t_batch[i])
        data_list.append(data)
    batch_data = Batch.from_data_list(data_list)
    return batch_data

# 定义模型
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

# 训练模型
def train_model(batch_data, model, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(batch_data)
        
        # Compute loss
        loss = 0
        batch_size = len(batch_data.batch.unique())
        for i in range(batch_size):
            batch_mask = batch_data.batch == i
            V_qd_new = out[batch_mask][0]
            cos_sim = F.cosine_similarity(V_qd_new.unsqueeze(0), batch_data.x[batch_mask].unsqueeze(0))
            criterion = torch.nn.CrossEntropyLoss()
            loss += criterion(cos_sim, torch.tensor([1.0]))
        
        loss = loss / batch_size
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# 示例数据
V_qd_batch = torch.randn(16, 1, 768)  # [batch_size, 1, 768]
V_s_batch = torch.randn(16, 4, 768)  # [batch_size, 4, 768]
V_t_batch = torch.randn(16, 4, 768)  # [batch_size, 4, 768]

batch_data = construct_batch(V_qd_batch.squeeze(1), V_s_batch, V_t_batch)
model = GCN(in_channels=768, out_channels=768)  # 根据需要选择 GCN 或 GAT
train_model(batch_data, model)
