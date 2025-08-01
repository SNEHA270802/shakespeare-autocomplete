from django.urls import path
from .views import index, suggest_words

urlpatterns = [
    path('', index, name='index'),
    path('api/suggest/', suggest_words, name='suggest_words'),
]
