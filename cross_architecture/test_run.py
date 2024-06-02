from pathlib import Path

import ir_datasets
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import ir_measures
from ir_measures import *

from argments import parse_args
from data2 import DatasetForTest, DataCollatorForTest

args = parse_args()

HOME_DIR = Path(__file__).parent.parent
model_path = str(HOME_DIR / 'models' / 'models--xlm-roberta-base')
tokenizer_path = str(HOME_DIR / 'models' / 'models--xlm-roberta-base')

# 加载测试数据
test1_dataset_obj = ir_datasets.load('clirmatrix/kk/bi139-base/zh/test1')
test1_qrels_df = pd.DataFrame(test1_dataset_obj.qrels_iter())

# 加载模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
model.to(device)

test_dataset = DatasetForTest(args=args)
test_datacollator = DataCollatorForTest(tokenizer, max_len=256)
test_dataloader = DataLoader(
    test_dataset, shuffle=False, collate_fn=test_datacollator, batch_size=args.per_device_test_batch_size
)

scores = []
for batch_idx, batch in enumerate(test_dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    # 要不要进行一个 softmax 操作？
    batch_socres = output.logits.squeeze(1).tolist()
    scores.extend(batch_socres)

if len(scores) != test1_qrels_df.shape[0]:
    ValueError('模型计算的test数据集的得分 数量上和原数据不等')

run_qrels_df = test1_qrels_df.drop(['relevance', 'iteration'], axis=1, inplace=False)
run_qrels_df['score'] = scores

METRICS_LIST = [R@5, R@10, RR@5, RR@10, nDCG@5, nDCG@10]
results = ir_measures.calc_aggregate(METRICS_LIST, test1_qrels_df, run_qrels_df)

test_results = {}
for metric in METRICS_LIST:
    test_results[str(metric)] = results[metric]

print(test_results)
