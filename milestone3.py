
# streamlit_app.py
import streamlit as st
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib

# Download required nltk data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load saved model and vectorizer
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocessing tools
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean HTML, lowercase, remove non-alpha, tokenize, remove stopwords, and lemmatize."""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Fake Job Posting Detector", page_icon="")

st.title(" Fake Job Posting Detector")
st.write("Paste a job description below to check if it’s *Real or Fraudulent*.")

# User Input
user_input = st.text_area("Enter Job Description:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning(" Please enter a job description.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        prob = model.predict_proba(vectorized_text)[0][prediction]

        if prediction == 1:
            st.error(f"This job posting looks *Fraudulent* (confidence: {prob:.2f})")
        else:
            st.success(f" This job posting looks *Real* (confidence: {prob:.2f})")
