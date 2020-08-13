from django.shortcuts import render
from .apps import PredictionappConfig
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
import torch
from captum.attr import LayerIntegratedGradients, TokenReferenceBase

class Sentiment_Model_Analyse(APIView):
    def post(self, request, format = None):
        # get input from request
        sentence = str(request.data['text'])
        # access loaded model config from apps.py
        device = PredictionappConfig.device
        tokenizer = PredictionappConfig.tokenizer
        preprocessor = PredictionappConfig.preprocessor
        model = PredictionappConfig.model
        # set to inference mode and send to device (cpu)
        model.eval()
        model = model.to(device)
        # reference token index = pad token index
        pad_index = tokenizer.vocab.stoi['pad']
        reference_token = TokenReferenceBase(reference_token_idx = pad_index)
        # initialize feature attribution model
        lig = LayerIntegratedGradients(model, model.embedding)
        # tokenize text
        text = [token.text for token in preprocessor.tokenizer(sentence)]
        # pad to minimum length
        min_len = 7
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
        # predict probability with model
        pred_prob = torch.sigmoid(model(indices_tensor)).item()
        # generate reference indices
        reference_indices = reference_token.generate_reference(len(text), device = device).unsqueeze(0)
        # compute attributions using layer-integrated gradients
        attributions = lig.attribute(
            indices_tensor,
            reference_indices,
            n_steps = 500
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