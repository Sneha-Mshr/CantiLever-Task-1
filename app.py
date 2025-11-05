# app.py
import streamlit as st
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('movie_reviews', quiet=True)

# --- prepare data
fileids = movie_reviews.fileids()
docs = [" ".join(movie_reviews.words(fid)) for fid in fileids]
labels = [1 if fid.startswith("pos/") else 0 for fid in fileids]

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X = vectorizer.fit_transform(docs)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# --- UI
st.title("🎭 Sentiment Analysis System")
st.caption(f"Model test accuracy: **{acc:.3f}**")

review = st.text_area("Enter a movie review:")

if st.button("Analyze"):
    if not review.strip():
        st.warning("Please enter some text!")
    else:
        x = vectorizer.transform([review])
        if x.nnz == 0:  # no known words
            st.info("⚠️ Your review used words unseen in training — try a longer sentence.")
        else:
            pred = model.predict(x)[0]
            prob = model.predict_proba(x)[0][pred]
            if pred == 1:
                st.success(f"✅ Sentiment: POSITIVE ({prob:.2f} confidence)")
            else:
                st.error(f"❌ Sentiment: NEGATIVE ({prob:.2f} confidence)")
