from django.apps import AppConfig
#######
#import pandas as pd
#from joblib import load
#######
import os
import spacy
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import torchtext.data
from torchtext import vocab
from torchtext.vocab import Vocab
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

# class CNN(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
#         self.convs = nn.ModuleList([
#             nn.Conv2d(1, n_filters, kernel_size = (fs, embedding_dim)) 
#             for fs
#             in filter_sizes
#         ])
#         self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, text):
#         embedded = self.embedding(text) # (n, T) -> (n, T, E)
#         embedded = embedded.unsqueeze(1) # (n, T, E) -> (n, 1, T, E)
#         conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs] # (n, 1, T, E) -> (n, n_filters, T - d_filter + 1)
#         pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] # (n, n_filters, T - d_filter + 1) -> (n, n_filters)
#         cat = self.dropout(torch.cat(pooled, dim = 1)) # (n, n_filters) -> (n, n_filters * len(filter_sizes))
#         logit = self.fc(cat)
#         return logit

class PredictionappConfig(AppConfig):
    name = 'PredictionApp'
    # set device
    device = torch.device('cpu')
    # load model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'PredictionApp/models/')
    # model = torch.load(os.path.join(model_dir, 'imdb-model-cnn.pt'))
    # load SpaCy processor
    spacy_processor = spacy.load('en_core_web_sm')
    # load TorchText tokenizer (IMDB vocabulary + GloVe embeddings)
    tokenizer = torch.load(os.path.join(model_dir, 'torchtext_tokenizer.pt'))