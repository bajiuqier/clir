import os
import torch
from torch import Tensor
from torch import nn
from transformers import AutoModel
from typing import Dict, List, Tuple, Any, Optional, NamedTuple
from argments import parse_args

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

    def forward(self, features: Dict[str, Tensor]):
        model_output = self.model(**features, return_dict=True)

        if self.pooling_method == 'cls':
            text_representation = model_output.last_hidden_state[:, 0, :]
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

class DualModel(nn.Module):
    def __init__(self, args: parse_args) -> None:
        super().__init__()
        self.pooling_method = args.pooling_method
        self.normalized = args.normalized
        self.embedding_model = EmbeddingModel(args=args)

        if self.training:
            self.loss_function = nn.CrossEntropyLoss()
        
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
            scores = self.compute_similarity(q_representation, d_representation)
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long) * 2
            loss = self.loss_function(scores, target)

        return OutputTuple(
            loss=loss,
            # query_vector=q_representation,
            # doc_vector=d_representation,
        )
    
    def save_model(self, output_dir: str):
        self.embedding_model.save_model(output_dir)
        model_status_save_path = os.path.join(output_dir, 'model.pth')
        torch.save(self.embedding_model.state_dict(), model_status_save_path)


class DualModel2(nn.Module):
    def __init__(self, args: parse_args) -> None:
        super().__init__()
        self.args = args
        self.model = AutoModel.from_pretrained(self.args.model_name_or_path)
        # self.normlized = self.args.normlized
        # self.pooling_method = self.args.pooling_method

        # 如果处于训练模式 定义损失函数
        if self.training:
            self.loss_function = nn.CrossEntropyLoss()
        
    def get_embedding(self, features = None):
        if features is None:
            return None
        
        model_output = self.model(**features, return_dict=True)

        if self.args.pooling_method == 'cls':
            sentence_representation = model_output.last_hidden_state[:, 0, :]
        if self.args.pooling_method == 'mean':
            # 这里暂时没有看懂
            attention_mask = features['attention_mask']
            s = torch.sum(model_output.last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(axis=1, keepdim=True).float()
            sentence_representation = s / d
        
        if self.args.normlized:
            sentence_representation = torch.nn.functional.normalize(sentence_representation, dim=-1)

        return sentence_representation.contiguous()
    
    def compute_similarity(self, q_representation: Tensor, p_representation: Tensor):
        if self.args.normlized:
            if len(p_representation.size()) == 2:
                return torch.matmul(q_representation, p_representation.transpose(0, 1))
            return torch.matmul(q_representation, p_representation.transpose(-2, -1))
        
        else:
            # q_reps q_reps 都应该是二维的
            return torch.cosine_similarity(q_representation.unsqueeze(1),
                                           p_representation.unsqueeze(0),
                                           dim=-1
                                           )
    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_representation = self.get_embedding(query)
        p_representation = self.get_embedding(passage)

        if not self.training:
            scores = self.compute_similarity(q_representation, p_representation)
            loss = None
        else:
            scores = self.compute_similarity(q_representation,p_representation)
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long) * 2
            loss = self.loss_function(scores, target)
        return OutputTuple(
            loss=loss,
            query_vector=q_representation,
            doc_vector=p_representation,
            # scores=scores,
        )
