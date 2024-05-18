import gzip
from pathlib import Path

data_path = Path.home() / "autodl-tmp" / "Datasets" / "CLIRMatrix" / "BI-139" / "full"
doc_data_path = Path.home() / "autodl-tmp" / "Datasets" / "CLIRMatrix" / "Document"

dev_file = data_path / "zh.kk.dev.jl.gz"
test1_file = data_path / "zh.kk.test1.jl.gz"
train_file = data_path / "zh.kk.train.jl.gz"
doc_file = doc_data_path / "kk.tsv.gz"

with gzip.open(train_file, 'rt', encoding='utf-8') as f:
    # 读取文本内容
    for _ in range(10) :
        line = f.readline()
        print(line)

with gzip.open(doc_file, 'rt', encoding='utf-8') as f:
    # 读取文本内容
    for _ in range(10) :
        line = f.readline()
        print(line)