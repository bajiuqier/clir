import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

HOME_DIR = Path.home().parent

# 指定保存模型的目录
cache_directory = "/path/to/save/model"
cache_directory = str(HOME_DIR / 'models' / 'xlm-roberta-base')

# 加载并保存模型和分词器到指定目录
model = AutoModel.from_pretrained("xlm-roberta-base", trust_remote_code=True, cache_dir=cache_directory)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", trust_remote_code=True, cache_dir=cache_directory)

print(model)