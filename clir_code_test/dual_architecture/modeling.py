import os
import torch
from torch import Tensor
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification
from typing import Dict, List, Tuple, Any, Optional, NamedTuple

from argments import parse_args
from criteria import CustomCosineEmbeddingLoss

class OutputTuple(NamedTuple):
    loss: Optional[torch.Tensor] = None
    query_vector: Optional[torch.Tensor] = None
    doc_vector: Optional[torch.Tensor] = None
    embedding: Optional[torch.Tensor] = None

class EmbeddingModel(nn.Module):
    def __init__(self, args: parse_args):
        super().__init__()
        self.model = AutoModel.from_pretrained(args.model_name_or_path)
        self.pooling_method = args.pooling_method
        self.normalized = args.normalized
        # self.linear = nn.Linear(self.model.config.hidden_size, 128)

    def forward(self, features: Dict[str, Tensor]):
        model_output = self.model(**features, return_dict=True)

        if self.pooling_method == 'cls':
            text_representation = model_output.last_hidden_state[:, 0, :]
            # text_representation = self.linear(text_representation)
        elif self.pooling_method == 'mean':
            attention_mask = features['attention_mask']
            s = torch.sum(model_output.last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(axis=1, keepdim=True).float()
            text_representation = s / d
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")

        if self.normalized:
            text_representation = torch.nn.functional.normalize(text_representation, dim=-1)
        
        text_representation = text_representation.contiguous()
        
        return OutputTuple(embedding=text_representation)
    
    def save_model(self, output_dir: str):
        self.model.save_pretrained(os.path.join(output_dir, "embedding_model"))

    @classmethod
    def from_pretrained(cls, model_path: str, args: parse_args):
        args.model_name_or_path = model_path
        model = cls(args)
        return model


class DualModel(nn.Module):
    def __init__(self, args: parse_args) -> None:
        super().__init__()
        self.pooling_method = args.pooling_method
        self.normalized = args.normalized
        self.embedding_model = EmbeddingModel(args=args)

        if self.training:
            # self.loss_function = nn.CrossEntropyLoss()
            self.loss_function = CustomCosineEmbeddingLoss()
        
    def compute_similarity(self, q_representation: Tensor, d_representation: Tensor):
        if self.normalized:
            if len(d_representation.size()) == 2:
                return torch.matmul(q_representation, d_representation.transpose(0, 1))
            return torch.matmul(q_representation, d_representation.transpose(-2, -1))
        else:
            return torch.cosine_similarity(q_representation.unsqueeze(1),
                                           d_representation.unsqueeze(0),
                                           dim=-1)

    def forward(self, query: Dict[str, Tensor] = None, document: Dict[str, Tensor] = None):
        q_representation = self.embedding_model(query).embedding
        d_representation = self.embedding_model(document).embedding

        if not self.training:
            scores = self.compute_similarity(q_representation, d_representation)
            loss = None
        else:
            # scores = self.compute_similarity(q_representation, d_representation)
            scores = self.compute_similarity(q_representation.unsqueeze(1), d_representation.view(q_representation.size(0), 2, -1)).squeeze(1)
            # 将 scores 重构为一个大小为 [batch内query的数量, 自动求出列的大小] 其实列的大小就是 一个query对应的正负样本的总和
            scores = scores.view(q_representation.size(0), -1)
            # target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
            target = torch.ones(scores.size(0), scores.size(1), device=scores.device, dtype=torch.int)
            target[:, 1] = -1
            loss = self.loss_function(scores, target)

        return OutputTuple(
            loss=loss,
            # query_vector=q_representation,
            # doc_vector=d_representation,
        )
    
    def save_model(self, output_dir: str):
        self.embedding_model.save_model(output_dir)
        # 使用 pytorch 保存 模型的静态参数字典
        model_status_save_path = os.path.join(output_dir, 'model.pth')
        torch.save(self.embedding_model.state_dict(), model_status_save_path)
