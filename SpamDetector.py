import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load dataset
data = pd.read_csv(
    r"C:\Users\91629\Downloads\spam.csv",
    encoding='latin1'
)

# Keep only required columns
data = data[['v1', 'v2']]

# Rename columns
data.columns = ['Category', 'Message']

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Convert labels
data['Category'] = data['Category'].replace({
    'ham': 'Not Spam',
    'spam': 'Spam'
})

# Input and output columns
mess = data['Message']
cat = data['Category']

# Split dataset
mess_train, mess_test, cat_train, cat_test = train_test_split(
    mess,
    cat,
    test_size=0.2,
    random_state=42
)

# Convert text into vectors
cv = CountVectorizer(stop_words='english')

features = cv.fit_transform(mess_train)

# Train model
model = MultinomialNB()
model.fit(features, cat_train)

# Prediction function
def predict(message):
    input_message = cv.transform([message])
    result = model.predict(input_message)
    return result[0]

# Streamlit UI
st.header('Spam Detection')

input_mess = st.text_input('Enter Message Here')

if st.button('Validate'):
    output = predict(input_mess)
    st.write(f'Prediction: {output}')