import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pickle

# Function to load emails and labels
def load_data(train_path, test_path):
    emails, labels = [], []
    
    # Load training data
    for filename in os.listdir(train_path):
        with open(os.path.join(train_path, filename), 'r', encoding='utf-8') as file:
            emails.append(file.read())
            labels.append(1 if 'spm' in filename else 0)  # 1 for spam, 0 for non-spam

    # Load testing data
    for filename in os.listdir(test_path):
        with open(os.path.join(test_path, filename), 'r', encoding='utf-8') as file:
            emails.append(file.read())
            labels.append(1 if 'spm' in filename else 0)  # 1 for spam, 0 for non-spam
            
    return pd.DataFrame({'text': emails, 'label': labels})

# Set the path to your dataset
train_path = r"C:\Users\subha\Downloads\train_test_mails\train-mails"
test_path = r"C:\Users\subha\Downloads\train_test_mails\test-mails"
df = load_data(train_path, test_path)

# Vectorize the text data
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['text'])
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models
models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

# Store the results
results = {}

# Train and evaluate each model with cross-validation
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # 5-fold cross-validation
    mean_cv_score = cv_scores.mean()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results[model_name] = {
        'accuracy': accuracy,
        'mean_cv_score': mean_cv_score,
        'classification_report': classification_report(y_test, y_pred)
    }
    
    print(f"{model_name} - Cross-Validation Score: {mean_cv_score:.4f}, Test Accuracy: {accuracy:.4f}")
    print(results[model_name]['classification_report'])

# Find the best model based on cross-validation score
best_model_name = max(results, key=lambda k: results[k]['mean_cv_score'])
best_model = models[best_model_name]

print(f"Best Model: {best_model_name} with Cross-Validation Score: {results[best_model_name]['mean_cv_score']:.4f}")

# Specify output directory
output_dir = "model"  # Change to your preferred path
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Save the best model and TF-IDF vectorizer
try:
    with open(os.path.join(output_dir, "best_model.pkl"), "wb") as model_file:
        pickle.dump(best_model, model_file)
    print("Best model saved successfully.")
    
    with open(os.path.join(output_dir, "tfidf_vectorizer.pkl"), "wb") as tfidf_file:
        pickle.dump(tfidf, tfidf_file)
    print("TF-IDF vectorizer saved successfully.")
except Exception as e:
    print(f"Error while saving files: {e}")
