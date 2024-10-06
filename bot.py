import json
import random
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load training data from JSON file
try:
    with open('training_data.json', 'r') as json_file:
        training_data = json.load(json_file)

    # Load testing data from JSON file
    with open('testing_data.json', 'r') as json_file:
        testing_data = json.load(json_file)

except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()

# Prepare training data
train_patterns = [entry.get('instruction (string)', '') for entry in training_data]
train_labels = [entry.get('response (string)', '') for entry in training_data]

# Convert training data to DataFrame for easier manipulation
train_df = pd.DataFrame({
    'patterns': train_patterns,
    'labels': train_labels
})

# Feature extraction
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['patterns'])
y_train = train_df['labels']

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prepare for chatbot interaction
st.title("Menstrual Health Chatbot")

# User input
user_input = st.text_input("You: ", "")

if user_input:
    # Vectorize user input
    X_user = vectorizer.transform([user_input])
    
    # Get prediction
    predicted_label = model.predict(X_user)
    
    # Display the response
    st.write(f"Chatbot: {predicted_label[0]}")

# Optionally, evaluate the model using testing data
if st.button("Evaluate Model"):
    test_patterns = [entry.get('instruction (string)', '') for entry in testing_data]
    test_labels = [entry.get('response (string)', '') for entry in testing_data]
    
    X_test = vectorizer.transform(test_patterns)
    y_test = test_labels

    # Predict on test data
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    st.write(f"Model Accuracy: {accuracy:.2f}")
