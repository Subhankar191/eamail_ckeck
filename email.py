import os
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

# Function to load emails and labels from a specified directory
def load_data(path):
    emails, labels = [], []
    
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
            emails.append(file.read())
            labels.append(1 if 'spm' in filename else 0)  # 1 for spam, 0 for non-spam
            
    return pd.DataFrame({'text': emails, 'label': labels})

# Set the path to your dataset
train_path = r"C:\Users\subha\Downloads\train_test_mails\train-mails"
test_path = r"C:\Users\subha\Downloads\train_test_mails\test-mails"

# Load training and test data
train_df = load_data(train_path)
X_train = train_df['text']
y_train = train_df['label']
test_df = load_data(test_path)
X_test = test_df['text']
y_test = test_df['label']

# Vectorize the text data
tfidf = TfidfVectorizer(stop_words='english')
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

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
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_cv_score = cv_scores.mean()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1_score = f1_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    
    results[model_name] = {
        'cross_val_score': mean_cv_score,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'f1_score': test_f1_score,
        'precision': test_precision,
    }
# Find and print the best model based on the highest test accuracy
best_model_name = max(results, key=lambda k: results[k]['test_accuracy'])
best_model_accuracy = results[best_model_name]['test_accuracy']
print(f"Best Model based on Test Accuracy: {best_model_name} with Test Accuracy: {best_model_accuracy:.4f}")
# Prepare data for plotting
model_names = list(results.keys())
cv_scores = [results[model]['cross_val_score'] for model in model_names]
train_accuracies = [results[model]['train_accuracy'] for model in model_names]
test_accuracies = [results[model]['test_accuracy'] for model in model_names]
f1_scores = [results[model]['f1_score'] for model in model_names]
precisions = [results[model]['precision'] for model in model_names]

# Set up the data for grouped bar chart
bar_width = 0.15
x = np.arange(len(model_names))

# Set up the matplotlib figure with specified color palette
plt.figure(figsize=(12, 6))
colors = ['#003f5c', '#ffa600', '#ff6361', '#bc5090', '#58508d']

# Create bars for each metric with smaller text
plt.bar(x - 2 * bar_width, cv_scores, width=bar_width, label='Cross-Validation Score', color=colors[0])
plt.bar(x - bar_width, train_accuracies, width=bar_width, label='Train Accuracy', color=colors[1])
plt.bar(x, test_accuracies, width=bar_width, label='Test Accuracy', color=colors[2])
plt.bar(x + bar_width, f1_scores, width=bar_width, label='F1 Score', color=colors[3])
plt.bar(x + 2 * bar_width, precisions, width=bar_width, label='Precision', color=colors[4])

# Adding labels, title with reduced font size
plt.xlabel('Models', fontsize=10)
plt.ylabel('Scores', fontsize=10)
plt.title('Model Comparison: Cross-Validation, Train Accuracy, Test Accuracy, F1 Score, and Precision', fontsize=12)
plt.xticks(x, model_names, fontsize=9)
plt.yticks(fontsize=9)
plt.ylim(0.9, 1.05)  # Scale the y-axis slightly above 1.0
plt.legend(fontsize=8)
plt.tight_layout()

# Show the plot
plt.show()
