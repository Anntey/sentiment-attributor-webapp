from django.urls import path
import PredictionApp.views as views

urlpatterns = [
    path('predict/', views.Sentiment_Model_Analyse.as_view(), name = 'predict'),
]