from django.apps import AppConfig
import os
import spacy
import torch
from PredictionApp.training.model import CNN

class PredictionappConfig(AppConfig):
    name = 'PredictionApp'
    # set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'PredictionApp/training/')
    # set device
    device = torch.device('cpu')
    # load model
    model = CNN()
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt'), map_location = device))
    model.eval()
    model = model.to(device)
    # load spacy preprocessor
    preprocessor = spacy.load('en_core_web_sm')
    # load torchtext tokenizer (vocabulary + embeddings)
    tokenizer = torch.load(os.path.join(model_dir, 'tokenizer.pt'))