import os
import pickle
from django.shortcuts import render
from django.http import JsonResponse

# Load the pre-trained model and vectorizer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'classifier/model/best_model.pkl')
tfidf_path = os.path.join(BASE_DIR, 'classifier/model/tfidf_vectorizer.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(tfidf_path, 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)





def index(request):
    return render(request, 'classifier/index.html')

def predict(request):
    if request.method == 'POST':
        email_text = request.POST.get('email_text')
        vectorized_text = tfidf_vectorizer.transform([email_text])
        prediction = model.predict(vectorized_text)

        return JsonResponse({'is_spam': bool(prediction[0])})
    return JsonResponse({'error': 'Invalid request'}, status=400)

