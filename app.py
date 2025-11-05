import streamlit as st
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# --- TRAIN MODEL (only first run takes time)
nltk.download('movie_reviews')

docs = [" ".join(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids()]
labels = [1 if "pos" in fileid else 0 for fileid in movie_reviews.fileids()]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# --- STREAMLIT UI
st.title("🎭 Sentiment Analysis System")

user_text = st.text_area("Enter a movie review:")

if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        x = vectorizer.transform([user_text])
        pred = model.predict(x)[0]

        if pred == 1:
            st.success("✅ Sentiment: POSITIVE")
        else:
            st.error("❌ Sentiment: NEGATIVE")
