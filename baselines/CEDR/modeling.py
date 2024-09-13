from typing import Dict, List, Tuple, Any, Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

from argments import add_model_args

class PairwiseHingeLoss(nn.Module):
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


class VanillaBertRanker(nn.Module):
    def __init__(self, model_args: add_model_args):
        super().__init__()
        self.mbert = BertModel.from_pretrained(model_args.model_name_or_path)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Linear(self.mbert.config.hidden_size, 1)
        if self.training:
            self.loss_function = PairwiseHingeLoss(margin=0.3, reduction="mean")

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.mbert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        scores = self.classifier(self.dropout(cls_output)).view(-1, 2)

        if not self.training:
            loss = None
        else:
            scores = self.softmax(scores)
            target = torch.tensor([1, 0], device=scores.device, dtype=torch.long).repeat(scores.size()[0], 1)

            loss = self.loss_function(scores, target)

        return OutputTuple(
            loss=loss,
            scores=scores
        )


class CedrPacrrRanker(nn.Module):
    def __init__(self, model_args: add_model_args, n_kernels=11, kernel_size=3, kmax=2):
        super().__init__()
        self.mbert = BertModel.from_pretrained(model_args.model_name_or_path)
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.kmax = kmax
        
        self.convs = nn.ModuleList([
            nn.Conv2d(self.bert.config.num_hidden_layers + 1, n_kernels, (i, i)) 
            for i in range(1, kernel_size + 1)
        ])
        
        conv_output_size = sum([n_kernels * kmax for _ in range(1, kernel_size + 1)])
        self.combine = nn.Linear(self.bert.config.hidden_size + conv_output_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.mbert(input_ids, attention_mask, token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Create similarity matrices
        all_layers = [outputs.hidden_states[0]] + list(outputs.hidden_states)
        simmat = torch.stack([self.sim_matrix(layer) for layer in all_layers], dim=1)
        
        # Apply PACRR-style convolutions
        conv_results = []
        for conv in self.convs:
            conv_result = conv(simmat)
            kmax_result = torch.topk(conv_result, k=self.kmax, dim=-1)[0]
            conv_results.append(kmax_result.view(kmax_result.size(0), -1))
        
        conv_features = torch.cat(conv_results, dim=-1)
        
        # Combine BERT and PACRR features
        combined_features = torch.cat([cls_output, conv_features], dim=-1)
        return self.combine(combined_features)

    def sim_matrix(self, layer):
        query_vecs = layer[:, :20, :]  # Assume first 20 tokens are query
        doc_vecs = layer[:, 20:, :]    # Remaining tokens are document
        query_norms = torch.norm(query_vecs, p=2, dim=-1, keepdim=True)
        doc_norms = torch.norm(doc_vecs, p=2, dim=-1, keepdim=True)
        return torch.bmm(query_vecs, doc_vecs.transpose(-1, -2)) / (query_norms * doc_norms.transpose(-1, -2))

class CedrKnrmRanker(nn.Module):
    def __init__(self, model_args: add_model_args, n_kernels=11):
        super().__init__()
        self.mbert = BertModel.from_pretrained(model_args.model_name_or_path)
        self.n_kernels = n_kernels
        self.mu = nn.Parameter(torch.tensor([-1.0 + 2.0 * i / (n_kernels - 1) for i in range(n_kernels)]))
        self.sigma = nn.Parameter(torch.tensor([0.1] * n_kernels))
        self.combine = nn.Linear(self.bert.config.hidden_size + n_kernels * (self.bert.config.num_hidden_layers + 1), 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.mbert(input_ids, attention_mask, token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        all_layers = [outputs.hidden_states[0]] + list(outputs.hidden_states)
        kernels = []
        for layer in all_layers:
            simmat = self.sim_matrix(layer)
            kernels.append(self.kernel_pooling(simmat))
        
        kernel_features = torch.cat(kernels, dim=-1)
        combined_features = torch.cat([cls_output, kernel_features], dim=-1)
        return self.combine(combined_features)

    def sim_matrix(self, layer):
        query_vecs = layer[:, :20, :]  # Assume first 20 tokens are query
        doc_vecs = layer[:, 20:, :]    # Remaining tokens are document
        return F.cosine_similarity(query_vecs.unsqueeze(2), doc_vecs.unsqueeze(1), dim=-1)

    def kernel_pooling(self, simmat):
        kernels = torch.exp(-0.5 * (simmat.unsqueeze(-1) - self.mu.view(1, 1, 1, -1))**2 / self.sigma.view(1, 1, 1, -1)**2)
        return kernels.sum(dim=[1, 2])

class CedrDrmmRanker(nn.Module):
    def __init__(self, model_args: add_model_args, n_bins=11, hidden_size=5):
        super().__init__()
        self.mbert = BertModel.from_pretrained(model_args.model_name_or_path)
        self.n_bins = n_bins
        self.hidden_size = hidden_size
        self.histogram = nn.Linear(n_bins, hidden_size)
        self.combine = nn.Linear(self.bert.config.hidden_size + hidden_size * (self.bert.config.num_hidden_layers + 1), 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.mbert(input_ids, attention_mask, token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        all_layers = [outputs.hidden_states[0]] + list(outputs.hidden_states)
        histograms = []
        for layer in all_layers:
            simmat = self.sim_matrix(layer)
            hist = self.histogram_pooling(simmat)
            histograms.append(self.histogram(hist))
        
        histogram_features = torch.cat(histograms, dim=-1)
        combined_features = torch.cat([cls_output, histogram_features], dim=-1)
        return self.combine(combined_features)

    def sim_matrix(self, layer):
        query_vecs = layer[:, :20, :]  # Assume first 20 tokens are query
        doc_vecs = layer[:, 20:, :]    # Remaining tokens are document
        return F.cosine_similarity(query_vecs.unsqueeze(2), doc_vecs.unsqueeze(1), dim=-1)

    def histogram_pooling(self, simmat):
        hist = torch.zeros(simmat.size(0), self.n_bins).to(simmat.device)
        bin_boundaries = torch.linspace(-1, 1, self.n_bins + 1).to(simmat.device)
        for i in range(self.n_bins):
            hist[:, i] = ((simmat >= bin_boundaries[i]) & (simmat < bin_boundaries[i+1])).float().sum(dim=[1,2])
        return hist / (simmat.size(1) * simmat.size(2))


# if __name__ == "__main__":

#     from data import DatasetForCEDR, DataCollatorForCEDR
#     from argments import add_logging_args, add_training_args
#     from torch.utils.data import DataLoader

#     logging_args = add_logging_args()
#     model_args = add_model_args()
#     training_args = add_training_args()

#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     VanillaBert = VanillaBertRanker(model_args=model_args).to(device)

#     tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path, clean_up_tokenization_spaces=False)

#     train_dataset = DatasetForMBERT(dataset_file=training_args.train_dataset_name_or_path, dataset_type='train')

#     train_data_collator = DataCollatorForMBERT(tokenizer, max_len=256, training=True)

#     train_dataloader = DataLoader(
#         train_dataset, shuffle=True, collate_fn=train_data_collator, batch_size=training_args.batch_size, drop_last=True
#     )


#     for batch_idx, batch in enumerate(train_dataloader):
        # qd_batch = {k: v.to(device) for k, v in batch['qd_batch'].items()}
        # input_ids = qd_batch['input_ids']
        # attention_mask = qd_batch['attention_mask']
        # token_type_ids = qd_batch['token_type_ids']

        # outputs = VanillaBert(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

#         loss = outputs.loss
#         print(loss)



    
