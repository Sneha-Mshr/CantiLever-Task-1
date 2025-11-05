# sentiment_analysis_improved.py

import nltk
import random
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# download dataset
nltk.download('movie_reviews')

# load movie review dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# shuffle to avoid order bias
random.shuffle(documents)

# convert list of words to sentences
X = [" ".join(words) for words, label in documents]
y = [label for words, label in documents]      # pos / neg

# TF-IDF + Bi-grams
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2)          # THIS captures “not good”
)

X_features = vectorizer.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42
)

# logistic regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# custom user review
while True:
    user_review = input("\nEnter your review (or 'q' to exit): ")
    if user_review.lower() == "q":
        break
    user_features = vectorizer.transform([user_review])
    prediction = model.predict(user_features)[0]
    print("Sentiment:", prediction)
