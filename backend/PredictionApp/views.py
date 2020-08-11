from django.shortcuts import render
from .apps import PredictionappConfig
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
#########
import pandas as pd
#########
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

class Add_Values(APIView):
    def post(self, request, format=None):
        sum = 0
        data = request.data
        for key in data:
            sum += data[key]
        response_dict = {"sum": sum}
        return Response(response_dict, status=status.HTTP_201_CREATED)

class IRIS_Model_Predict(APIView):
    #permission_classes = [IsAuthenticated]
    #throttle_classes = [LimitedRateThrottle]
    def post(self, request, format=None):
        data = request.data
        keys = []
        values = []
        for key in data:
            keys.append(key)
            values.append(data[key])
        X = pd.Series(values).to_numpy().reshape(1, -1)
        loaded_classifier = PredictionappConfig.model
        y_pred = loaded_classifier.predict(X)
        y_pred = pd.Series(y_pred)
        target_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        y_pred = y_pred.map(target_map).to_numpy()
        response_dict = {"Prediced Iris Species": y_pred[0]}
        return Response(response_dict, status=200)

class Sentiment_Model_Analyse(APIView):
    def post(self, request, format = None):
        # parse input from request
        sentence = request.data['text']
        # set argument
        min_len = 7
        # access loaded model config from apps.py
        device = PredictionappConfig.device
        #model = PredictionappConfig.model

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_file = os.path.join(base_dir, 'PredictionApp/models/imdb-model-cnn.pt')
        model = torch.load(model_file)
        print('loaded model')
        model.eval()
        model = model.to(device)

        tokenizer = PredictionappConfig.tokenizer
        spacy_processor = PredictionappConfig.spacy_processor
        # reference/baseline token index = pad token index
        pad_index = tokenizer.vocab.stoi['pad']
        token_reference = TokenReferenceBase(reference_token_idx = pad_index)
        # initialize feature attribution model
        lig = LayerIntegratedGradients(model, model.embedding)
        # tokenize text
        text = [token.text for token in spacy_processor.tokenizer(sentence)]
        # pad to minimum length
        if len(text) < min_len:
            text += ['pad'] * (min_len - len(text))
        # get indices for token strings
        indices = [tokenizer.vocab.stoi[token] for token in text]
        # clear gradients
        model.zero_grad()
        # initialize input tensor
        indices_tensor = torch.tensor(indices, device = device)
        # induce batch dim
        indices_tensor = indices_tensor.unsqueeze(0)
        # save input seuqnce length
        seq_length = len(text) #min_len
        # predict probability with model
        pred_prob = torch.sigmoid(model(indices_tensor)).item()
        # generate reference indices
        reference_indices = token_reference.generate_reference(seq_length, device = device).unsqueeze(0)
        # compute attributions using layer-integrated gradients
        attributions, delta = lig.attribute(
            indices_tensor,
            reference_indices,
            n_steps = 500,
            return_convergence_delta = True
        )
        # sum attributions along embedding dimensions and norm to [-1, 1]
        attributions = attributions.sum(dim = 2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        # construct response json object
        response_json = {
            'prob': pred_prob,
            'attributions': attributions,
            'text': text
        }
        # return probability, attributions and text
        return Response(response_json, status = 200)