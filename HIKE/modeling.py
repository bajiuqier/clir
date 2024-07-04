import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import ir_datasets

from pathlib import Path


class CLIRDataset(Dataset):
    def __init__(self, queries, documents, labels, kg_data, tokenizer, max_len=512):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.kg_data = kg_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        document = self.documents[idx]
        label = self.labels[idx]
        kg_info = self.kg_data[idx]

        encoded_pair = self.tokenizer.encode_plus(
            query,
            document,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        encoded_kg = self.tokenizer.encode_plus(
            kg_info,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded_pair['input_ids'].flatten(),
            'attention_mask': encoded_pair['attention_mask'].flatten(),
            'kg_input_ids': encoded_kg['input_ids'].flatten(),
            'kg_attention_mask': encoded_kg['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    # @staticmethod
    # def 
    
class KnowledgeLevelFusion(nn.Module):
    def __init__(self, hidden_size: int=768, num_heads: int=8, dropout=0.1) -> None:
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        output = self.multihead_attn(Q, K, V)
        E_KG = self.fc(output)
        E_KG = self.tanh(E_KG)
        E_KG = self.dropout(E_KG)

        return E_KG

class LanguageLevelFusion(nn.Module):
    def __init__(self, hidden_size=768) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, V_qd, E_s_KG, E_t_KG):
        Vector = torch.cat([V_qd, E_s_KG, E_t_KG], dim=0)
        output = self.fc(Vector)
        output = self.tanh(output)

        return output

class HIKE(nn.Module):
    def __init__(self, mbert_model_name, hidden_size=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.knowledge_level_fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.language_level_fusion = nn.Linear(hidden_size * 3, hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)

        self.mbert = BertModel.from_pretrained(mbert_model_name)
        self.multihead_attn = nn.MultiheadAttention(self.mbert.config.hidden_size, num_heads, dropout=dropout)
        self.fc = nn.Linear(self.mbert.config.hidden_size, 2)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, qd_batch, s_kg_batch, t_kg_batch):
        qd_output = self.mbert(**qd_batch)
        s_kg_output = self.mbert(**s_kg_batch)
        t_kg_output = self.mbert(**t_kg_batch)

        V_qd = qd_output.last_hidden_state[:, 0, :]
        E_s_kg = s_kg_output.last_hidden_state[:, 0, :]
        E_t_kg = t_kg_output.last_hidden_state[:, 0, :]


        # Knowledge-level fusion
        attn_output, _ = self.multi_head_attention(query_doc_embedding.unsqueeze(0),
                                                   kg_embedding.unsqueeze(0),
                                                   kg_embedding.unsqueeze(0))
        knowledge_fused = self.knowledge_level_fusion(torch.cat([query_doc_embedding, attn_output.squeeze(0)], dim=-1))

        # Language-level fusion
        language_fused = self.language_level_fusion(torch.cat([query_doc_embedding, kg_embedding, knowledge_fused], dim=-1))

        # Classification
        logits = self.classifier(language_fused)
        return logits

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        kg_input_ids = batch['kg_input_ids'].to(device)
        kg_attention_mask = batch['kg_attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, kg_input_ids, kg_attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def main():
    # 设置参数
    batch_size = 32
    num_epochs = 10
    learning_rate = 2e-5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    # 这里需要根据实际情况加载您的数据
    train_dataset = CLIRDataset(train_queries, train_documents, train_labels, train_kg_data, tokenizer)
    # eval_dataset = CLIRDataset(eval_queries, eval_documents, eval_labels, eval_kg_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    # 初始化模型
    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
    model = HIKE(bert_model).to(device)

    # 定义优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        # eval_loss, eval_accuracy = evaluate(model, eval_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        # print(f'Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}')

if __name__ == '__main__':
    main()
