import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #e2e8f0;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #94a3b8;
        margin-bottom: 30px;
    }

    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #d1d5db;
        padding: 10px;
        font-size: 16px;
    }

    .result-box {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }

    .spam {
        background-color: #fee2e2;
        color: #b91c1c;
    }

    .not-spam {
        background-color: #dcfce7;
        color: #166534;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------
data = pd.read_csv("spam.csv", encoding='latin1')

# Keep only required columns
data = data[['v1', 'v2']]

# Rename columns
data.columns = ['Category', 'Message']

# Remove duplicates
data.drop_duplicates(inplace=True)

# Convert labels
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# ---------------- TRAIN MODEL ----------------
X = data['Message']
y = data['Category']

cv = CountVectorizer()
X = cv.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# ---------------- UI ----------------
st.markdown('<div class="title">📧 SMS Spam Classifier</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Detect whether a message is Spam or Not Spam using Machine Learning and NLP</div>',
    unsafe_allow_html=True
)

user_input = st.text_area(
    "Enter your message below:",
    height=150,
    placeholder="Example: Congratulations! You won a free iPhone..."
)

# ---------------- PREDICTION ----------------
if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed_input = cv.transform([user_input])
        prediction = model.predict(transformed_input)[0]

        if prediction == 'Spam':
            st.markdown(
                '<div class="result-box spam">🚨 Spam Message</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box not-spam">✅ Not Spam</div>',
                unsafe_allow_html=True
            )


