import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

HOME_DIR = Path.home() / 'Desktop' / 'clir'

# 指定保存模型的目录
model_path = str(HOME_DIR / 'models' / 'models--xlm-roberta-base')

# 加载并保存模型和分词器到指定目录
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(model)