import os
import torch
from torch import Tensor
from torch import nn
from transformers import AutoModelForSequenceClassification
from typing import Optional, NamedTuple

from argments import parse_args
from criteria import PairwiseSoftmaxLoss

class OutputTuple(NamedTuple):
    loss: Optional[torch.Tensor] = None
    query_vector: Optional[torch.Tensor] = None
    doc_vector: Optional[torch.Tensor] = None
    embedding: Optional[torch.Tensor] = None

class CrossModel(nn.Module):
    def __init__(self, args: parse_args) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=1)
        self.batch_size = args.per_device_train_batch_size
        if self.training:
            # self.loss_function = nn.CrossEntropyLoss()
            self.loss_function = PairwiseSoftmaxLoss()
        
    def get_scores(self, batch):
        logits = self.model(**batch).logits
        scores = logits.view(-1, 2)
        return scores

    def forward(self, batch):

        if not self.training:
            scores = self.get_scores(batch)
            loss = None
        else:
            scores = self.get_scores(batch)
            # target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
            # loss = self.loss_function(scores, target)
            loss = self.loss_function(scores)

        return OutputTuple(
            loss=loss,
            # query_vector=q_representation,
            # doc_vector=d_representation,
        )
    
    def save_model(self, output_dir: str):
        self.model.save_pretrained(os.path.join(output_dir, "cross_model"))
        # 使用 pytorch 保存 模型的静态参数字典
        model_status_save_path = os.path.join(output_dir, 'model.pth')
        torch.save(self.model.state_dict(), model_status_save_path)
