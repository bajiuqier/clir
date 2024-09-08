import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv

def construct_graph(V_qd, V_s, V_t):
    x = torch.cat([V_qd.unsqueeze(0), V_s, V_t], dim=0)
    edge_index = []

    for i in range(1, V_s.size(0) + 1):
        edge_index.append([0, i])
        edge_index.append([i, 0])

    for i in range(V_s.size(0) + 1, x.size(0)):
        edge_index.append([0, i])
        edge_index.append([i, 0])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
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
batch_size = 8
V_qd_batch = [torch.randn(768) for _ in range(batch_size)]
V_s_batch = [torch.randn(10, 768) for _ in range(batch_size)]
V_t_batch = [torch.randn(10, 768) for _ in range(batch_size)]

data_list = [construct_graph(V_qd, V_s, V_t) for V_qd, V_s, V_t in zip(V_qd_batch, V_s_batch, V_t_batch)]
batch_data = Batch.from_data_list(data_list)

model = GCN(in_channels=768, out_channels=768)  # 根据需要选择 GCN 或 GAT
train_model(batch_data, model)
