from django.urls import path
import PredictionApp.views as views

urlpatterns = [
    path('add_values/', views.Add_Values.as_view(), name = 'api_add_values'),
]