from django.apps import AppConfig

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

class PredictionappConfig(AppConfig):
    name = 'PredictionApp'
    # save paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'PredictionApp/models/')
    # set device
    device = torch.device('cpu')
    # load model
    model = torch.load(os.path.join(model_dir, 'imdb-model-cnn.pt'))
    # load SpaCy processor
    spacy_processor = spacy.load('en_core_web_sm')
    # load TorchText tokenizer (IMDB vocabulary + GloVe embeddings)
    tokenizer = torch.load(os.path.join(model_dir, 'torchtext_tokenizer.pt'))