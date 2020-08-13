import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size = 101982, embedding_dim = 50, n_filters = 100, filter_sizes = [3, 4, 5], output_dim = 1, dropout = 0.5, pad_idx = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, kernel_size = (fs, embedding_dim)) 
            for fs
            in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.embedding(text) # (n, T) -> (n, T, E)
        embedded = embedded.unsqueeze(1) # (n, T, E) -> (n, 1, T, E)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs] # (n, 1, T, E) -> (n, n_filters, T - d_filter + 1)
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] # (n, n_filters, T - d_filter + 1) -> (n, n_filters)
        cat = self.dropout(torch.cat(pooled, dim = 1)) # (n, n_filters) -> (n, n_filters * len(filter_sizes))
        logit = self.fc(cat)
        return logit