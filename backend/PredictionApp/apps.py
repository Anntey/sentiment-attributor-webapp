from django.apps import AppConfig
#######
import pandas as pd
from joblib import load
import os
#######

class PredictionappConfig(AppConfig):
    name = 'PredictionApp'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_folder = os.path.join(base_dir, 'PredictionApp/models/')
    model_file = os.path.join(model_folder, "iris_model.joblib")
    model = load(model_file)