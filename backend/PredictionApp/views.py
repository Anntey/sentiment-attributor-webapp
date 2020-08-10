from django.shortcuts import render
from .apps import PredictionappConfig
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
#########
import pandas as pd
#########

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

class Sentiment_Model_Predict(APIView):
    def post(self, request, format=None):
        data = request.data
        keys = []
        values = []
        for key in data:
            keys.append(key)
            values.append(data[key])
        X = pd.Series(values).to_numpy().reshape(1, -1)
        model = PredictionappConfig.model
        preds = model.predict(X)
        preds = pd.Series(y_pred)
        target_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        y_pred = y_pred.map(target_map).to_numpy()
        response_dict = {"Prediced Iris Species": y_pred[0]}
        return Response(response_dict, status=200)