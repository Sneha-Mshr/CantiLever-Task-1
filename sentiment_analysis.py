# sentiment_analysis.py

import nltk
import random
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download NLTK movie review dataset
nltk.download('movie_reviews')

# Load the dataset (words + labels)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle data to avoid bias
random.shuffle(documents)

# Convert list of words to string sentences
X = [" ".join(words) for words, label in documents]
y = [label for words, label in documents]  # 'pos' or 'neg'

# Convert text to feature vectors (Bag of Words)
vectorizer = CountVectorizer()
X_features = vectorizer.fit_transform(X)

#  Split into training and testing sets (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42)

#  Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions and check accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Try your own review
sample_review = ["The movie was boring and too long."]

user_review = input("Enter your movie review: ")
sample_features = vectorizer.transform([user_review])
prediction = model.predict(sample_features)[0]
print("Your Review Sentiment:", prediction)
