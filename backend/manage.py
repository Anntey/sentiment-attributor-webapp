#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
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


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'MainApp.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
