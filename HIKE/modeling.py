import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np

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
    
class HIKE(nn.Module):
    def __init__(self, bert_model, hidden_size=768, num_heads=6):
        super(HIKE, self).__init__()
        self.bert = bert_model
        self.multi_head_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.knowledge_level_fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.language_level_fusion = nn.Linear(hidden_size * 3, hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, kg_input_ids, kg_attention_mask):
        query_doc_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        kg_output = self.bert(input_ids=kg_input_ids, attention_mask=kg_attention_mask)

        query_doc_embedding = query_doc_output.last_hidden_state[:, 0, :]
        kg_embedding = kg_output.last_hidden_state[:, 0, :]

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
