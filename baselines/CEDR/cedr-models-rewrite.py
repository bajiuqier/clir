import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class BertRanker(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, token_type_ids):
        raise NotImplementedError

    def encode_bert(self, input_ids, attention_mask, token_type_ids):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

class VanillaBertRanker(BertRanker):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.encode_bert(input_ids, attention_mask, token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))

class CedrPacrrRanker(BertRanker):
    def __init__(self, model_name='bert-base-uncased', n_kernels=11, kernel_size=3, kmax=2):
        super().__init__(model_name)
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
        outputs = self.encode_bert(input_ids, attention_mask, token_type_ids)
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

class CedrKnrmRanker(BertRanker):
    def __init__(self, model_name='bert-base-uncased', n_kernels=11):
        super().__init__(model_name)
        self.n_kernels = n_kernels
        self.mu = nn.Parameter(torch.tensor([-1.0 + 2.0 * i / (n_kernels - 1) for i in range(n_kernels)]))
        self.sigma = nn.Parameter(torch.tensor([0.1] * n_kernels))
        self.combine = nn.Linear(self.bert.config.hidden_size + n_kernels * (self.bert.config.num_hidden_layers + 1), 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.encode_bert(input_ids, attention_mask, token_type_ids)
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

class CedrDrmmRanker(BertRanker):
    def __init__(self, model_name='bert-base-uncased', n_bins=11, hidden_size=5):
        super().__init__(model_name)
        self.n_bins = n_bins
        self.hidden_size = hidden_size
        self.histogram = nn.Linear(n_bins, hidden_size)
        self.combine = nn.Linear(self.bert.config.hidden_size + hidden_size * (self.bert.config.num_hidden_layers + 1), 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.encode_bert(input_ids, attention_mask, token_type_ids)
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

class CustomBertModel(AutoModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        return outputs.hidden_states
