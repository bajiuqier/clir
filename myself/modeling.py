import torch
from torch import Tensor
from torch import nn
from transformers import  AutoModel, AutoTokenizer
from transformers.file_utils import ModelOutput
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from argments import parse_args

@dataclass
class EncoderOutput(ModelOutput):
    query_vector: Optional[Tensor] = None
    doc_vector: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    # scores: Optional[Tensor] = None

class DaulModel(nn.Module):
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

        # output_dict = {'loss': loss, 'query_vector': q_representation, 'doc_vector': p_representation}
        return EncoderOutput(
            loss=loss,
            query_vector=q_representation,
            doc_vector=p_representation,
            # scores=scores,
        )
