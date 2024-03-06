import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, average_precision_score, ndcg_score
import numpy as np

# Step 1: 加载并预处理数据

class IRDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query_id, query_text = self.data.iloc[idx]
        doc_texts = self.data[self.data['query_id'] == query_id]['text'].tolist()
        labels = self.data[self.data['query_id'] == query_id]['relevance'].tolist()

        encoding = self.tokenizer(
            text=query_text,
            text_pair=doc_texts,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels, dtype=torch.float32)
        }

# Step 2: 构建模型

model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)

# Step 3: 定义训练和评估函数

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def train_model(model, train_dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

def evaluate_model(model, eval_dataloader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)

    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')
    average_precision = average_precision_score(np.array(all_labels), outputs.logits.softmax(dim=1).cpu().numpy(), average='macro')
    relevance_scores = np.array([outputs.logits.softmax(dim=1).cpu().numpy()[i][int(all_labels[i])] for i in range(len(all_labels))])
    ndcg_at_10 = ndcg_score(np.array([all_labels]), np.array([relevance_scores]), k=10)

    return accuracy, recall, average_precision, ndcg_at_10

# Step 4: 训练模型并进行评估

train_data_path = "path/to/train.csv"
eval_data_path = "path/to/eval.csv"

train_dataset = IRDataset(train_data_path, tokenizer)
eval_dataset = IRDataset(eval_data_path, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 5
for epoch in range(num_epochs):
    loss = train_model(model, train_dataloader, optimizer, criterion)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

accuracy, recall, average_precision, ndcg_at_10 = evaluate_model(model, eval_dataloader)

print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"MAP: {average_precision:.4f}")
print(f"NDCG@10: {ndcg_at_10:.4f}")
