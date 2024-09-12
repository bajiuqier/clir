# Import necessary libraries
import os
import argparse
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class RetrievalDataset(Dataset):
    def __init__(self, queries: Dict[str, str], docs: Dict[str, str], pairs: List[Tuple[str, str]], qrels: Dict[str, Dict[str, int]]):
        self.queries = queries
        self.docs = docs
        self.pairs = pairs
        self.qrels = qrels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        qid, did = self.pairs[idx]
        query = self.queries[qid]
        doc = self.docs[did]
        relevance = self.qrels.get(qid, {}).get(did, 0)
        return {
            'qid': qid,
            'did': did,
            'query': query,
            'document': doc,
            'relevance': relevance
        }

class RetrievalCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        queries = [item['query'] for item in batch]
        docs = [item['document'] for item in batch]

        encoded_inputs = self.tokenizer(
            queries,
            docs,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'qids': [item['qid'] for item in batch],
            'dids': [item['did'] for item in batch],
            'input_ids': encoded_inputs['input_ids'],
            'attention_mask': encoded_inputs['attention_mask'],
            'token_type_ids': encoded_inputs['token_type_ids'],
            'relevance': torch.tensor([item['relevance'] for item in batch], dtype=torch.float)
        }

class CedrModel(torch.nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        score = self.classifier(cls_output).squeeze(-1)
        return score

def train(model, train_dataloader, valid_dataloader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_valid_score = float('-inf')

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{args.num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            relevance = batch['relevance'].to(device)

            scores = model(input_ids, attention_mask, token_type_ids)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, relevance)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{args.num_epochs}, Average Loss: {avg_loss:.4f}')

        valid_score = validate(model, valid_dataloader, device)
        print(f'Validation Score: {valid_score:.4f}')

        if valid_score > best_valid_score:
            best_valid_score = valid_score
            torch.save(model.state_dict(), args.model_save_path)
            print(f'New best validation score! Saved model to {args.model_save_path}')

def validate(model, dataloader, device):
    model.eval()
    total_score = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            scores = model(input_ids, attention_mask, token_type_ids)
            total_score += scores.sum().item()

    return total_score / len(dataloader.dataset)

def read_data(query_file, doc_file, qrels_file, pairs_file):
    queries = {}
    docs = {}
    with open(query_file, 'r') as f:
        for line in f:
            qid, query = line.strip().split('\t')
            queries[qid] = query
    
    with open(doc_file, 'r') as f:
        for line in f:
            did, doc = line.strip().split('\t')
            docs[did] = doc

    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            qid, _, did, relevance = line.strip().split()
            qrels.setdefault(qid, {})[did] = int(relevance)

    pairs = []
    with open(pairs_file, 'r') as f:
        for line in f:
            qid, did = line.strip().split()
            pairs.append((qid, did))

    return queries, docs, qrels, pairs

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    queries, docs, qrels, train_pairs = read_data(args.query_file, args.doc_file, args.qrels_file, args.train_pairs_file)
    _, _, _, valid_pairs = read_data(args.query_file, args.doc_file, args.qrels_file, args.valid_pairs_file)

    train_dataset = RetrievalDataset(queries, docs, train_pairs, qrels)
    valid_dataset = RetrievalDataset(queries, docs, valid_pairs, qrels)

    collator = RetrievalCollator(tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = CedrModel(args.model_name)

    train(model, train_dataloader, valid_dataloader, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CEDR model')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Pretrained model name')
    parser.add_argument('--query_file', type=str, required=True, help='Path to query file')
    parser.add_argument('--doc_file', type=str, required=True, help='Path to document file')
    parser.add_argument('--qrels_file', type=str, required=True, help='Path to qrels file')
    parser.add_argument('--train_pairs_file', type=str, required=True, help='Path to train pairs file')
    parser.add_argument('--valid_pairs_file', type=str, required=True, help='Path to validation pairs file')
    parser.add_argument('--model_save_path', type=str, default='cedr_model.pt', help='Path to save the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')

    args = parser.parse_args()
    main(args)
