# Email Spam Classification using NLP
## Overview
This project is an Email Spam Classification System built using Machine Learning and Natural Language Processing (NLP). The system classifies emails/messages as **Spam** or **Not Spam**.
The model is trained using a labeled dataset and uses text vectorization techniques along with the Naive Bayes algorithm for prediction.

## Features
* Detects spam messages automatically
* Uses NLP techniques for text processing
* Machine Learning based classification
* Simple and interactive Streamlit web interface
* Fast and accurate predictions

## Technologies Used
* Python
* Pandas
* Scikit-learn
* NLP (CountVectorizer)
* Streamlit

## Dataset
The dataset used contains labeled SMS/email messages categorized as:
* Spam
* Ham (Not Spam)

Dataset file:
`spam.csv`

## Machine Learning Workflow
1. Load dataset
2. Data preprocessing
3. Remove duplicates
4. Convert labels into numeric/text form
5. Text vectorization using CountVectorizer
6. Train-test split
7. Train model using Multinomial Naive Bayes
8. Predict spam or non-spam messages

## Project Structure
```bash
├── app.py
├── spam.csv
├── README.md
```

## Sample Prediction
Input:
```text
Congratulations! You have won a free iPhone.
```
Output:
```text
Spam
```

## Future Improvements
* Use advanced NLP techniques like TF-IDF
* Improve accuracy using Deep Learning models
* Add email integration
* Deploy the project online

## Author
Ananya Priya
