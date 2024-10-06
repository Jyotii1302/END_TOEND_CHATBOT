import json
import pandas as pd
import requests
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Streamlit title
st.title("Menstrual Health Chatbot Training")

# Load training data from JSON file hosted on GitHub
url = 'https://raw.githubusercontent.com/Jyotii1302/END_TOEND_CHATBOT/main/training_data.json'
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    training_data = response.json()
else:
    st.error(f"Failed to load data: {response.status_code}")
    st.stop()

# Prepare training data
train_patterns = [entry['instruction (string)'] for entry in training_data]
train_labels = [entry['output (string)'] for entry in training_data]

# Check if there are empty labels and report unique labels
if not train_labels:
    st.error("No labels found in the training data.")
    st.stop()

unique_labels = set(train_labels)
st.write(f"Unique labels found: {unique_labels}")

# Convert training data to DataFrame for easier manipulation
train_df = pd.DataFrame({
    'patterns': train_patterns,
    'labels': train_labels
})

# Check if there are at least two classes
if len(unique_labels) < 2:
    st.error("Not enough unique classes in the training data. Please provide data with at least two different labels.")
    st.stop()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_df['patterns'], train_df['labels'], test_size=0.2, random_state=42)

# Vectorize the training data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Initialize the model
model = LogisticRegression()

# Fit the model
model.fit(X_train_vectorized, y_train)

# Vectorize the test data
X_test_vectorized = vectorizer.transform(X_test)

# Predict on the test data
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy on the Streamlit app
st.write(f"Model accuracy: {accuracy:.2f}")
