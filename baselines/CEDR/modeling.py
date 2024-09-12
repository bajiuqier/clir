import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AutoModelForSequenceClassification
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, NamedTuple
from pathlib import Path
from argments import add_model_args


class PairwiseHingeLoss(torch.nn.Module):
    def __init__(self, margin=0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, score, target):
        # Compute the loss based on the target labels
        loss = torch.where(target == 1, 1 - score, 
                           torch.clamp(score - self.margin, min=0))
        
        # Apply the reduction method
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

class OutputTuple(NamedTuple):
    loss: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    # query_vector: Optional[torch.Tensor] = None
    # doc_vector: Optional[torch.Tensor] = None
    # embedding: Optional[torch.Tensor] = None

class CrossMBERT(nn.Module):
    def __init__(self, model_args: add_model_args):
        super().__init__()
        # self.batch_size = 8
        self.encoder = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=1)
        self.hidden_size = self.encoder.config.hidden_size
        self.softmax = nn.Softmax(dim=1)
        if self.training:
            self.loss_function = PairwiseHingeLoss()

    def forward(self, qd_batch):

        logits = self.encoder(**qd_batch).logits
        scores = logits.view(-1, 2)

        if not self.training:
            loss = None
        else:
            scores = self.softmax(scores)
            # target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
            # loss = self.loss_function(scores, target)
            target = torch.tensor([1, 0], device=scores.device, dtype=torch.long).repeat(scores.size()[0], 1)
            loss = self.loss_function(scores, target)

        return OutputTuple(
            loss=loss,
            scores=scores
        )
