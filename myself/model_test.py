import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from data import MyDataset, DataCollatorForMe

batch_size = 8
dataset_file = str(Path(__file__).parent / 'data' / 'dataset.jsonl')
model_path = str(Path(__file__).parent.parent / 'models' / 'models--bert-base-multilingual-uncased')

encoder = BertModel.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

dataset = MyDataset(dataset_file=dataset_file)
data_collator = DataCollatorForMe(tokenizer, max_len=256)

train_dataloader = DataLoader(
    dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
)
multihead_attn = nn.MultiheadAttention(embed_dim=encoder.config.hidden_size, num_heads=6, dropout=0.1, batch_first=True)
linear1 = nn.Linear(5 * 768, 768)
linear2 = nn.Linear(3 * 768, 768)
linear3 = nn.Linear(2 * 768, 1)


tanh = nn.Tanh()
for batch_index, batch in enumerate(train_dataloader):
    # print(batch)
    qd_output = encoder(**batch['qd_batch'])
    ed_s_output = encoder(**batch['ed_s_batch'])
    ed_t_output = encoder(**batch['ed_t_batch'])

    V_qd = qd_output.last_hidden_state[:, 0, :]
    V_ed_s = ed_s_output.last_hidden_state[:, 0, :]
    V_ed_t = ed_t_output.last_hidden_state[:, 0, :]

    tensor1 = V_qd.unsqueeze(1)
    tensor2 = V_ed_s.view(8, 4, 768).repeat_interleave(2, dim=0)
    tensor3 = V_ed_t.view(8, 4, 768).repeat_interleave(2, dim=0)
    # tensor2_repeated = tensor2.repeat_interleave(2, dim=0)
    E_r_s = torch.cat((tensor1, tensor2), dim=1)
    E_r_t = torch.cat((tensor1, tensor3), dim=1)

    attn_output1, _ = multihead_attn(E_r_s, E_r_s, E_r_s)
    attn_output2, _ = multihead_attn(E_r_t, E_r_t, E_r_t)

    E_kg_s = tanh(linear1(attn_output1.reshape(16, 5 * 768)))
    E_kg_t = tanh(linear1(attn_output2.reshape(16, 5 * 768)))

    E_kg_lang = tanh(linear2(torch.cat((V_qd, E_kg_s, E_kg_t), dim=1)))




