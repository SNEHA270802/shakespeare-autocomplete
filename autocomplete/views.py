

import pickle
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.metrics.pairwise import cosine_similarity

# Load model and data
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('vocab_list.pkl', 'rb') as f:
    vocabulary = pickle.load(f)

# Compute vectors once
tfidf_matrix = vectorizer.transform(vocabulary)

def index(request):
    return render(request, 'autocomplete.html')

@csrf_exempt
def suggest_words(request):
    query = request.GET.get('query', '').lower()
    if not query:
        return JsonResponse({'suggestions': []})
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-5:][::-1]
    suggestions = [vocabulary[i] for i in top_indices if scores[i] > 0.1]
    return JsonResponse({'suggestions': suggestions})
