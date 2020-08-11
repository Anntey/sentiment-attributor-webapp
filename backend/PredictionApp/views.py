from django.shortcuts import render
from .apps import PredictionappConfig
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

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

class Sentiment_Model_Analyse(APIView):
    def post(self, request, format = None):
        # get input from request
        sentence = request.data['text']
        # set argument
        min_len = 7
        # access loaded model config from apps.py
        device = PredictionappConfig.device
        tokenizer = PredictionappConfig.tokenizer
        spacy_processor = PredictionappConfig.spacy_processor
        model = PredictionappConfig.model
        model.eval()
        model = model.to(device)
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