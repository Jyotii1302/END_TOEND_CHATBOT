import streamlit as st
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load training data from JSON file
with open(r'C:\Users\User\NLP\training_data.json', 'r') as json_file:
    training_data = json.load(json_file)

# Check if training data is empty
if not training_data:
    st.error("Training data is empty. Please check the training data source.")
    st.stop()  # Stop execution if there's no training data

# Prepare training data
train_patterns = [entry.get('instruction (string)', '') for entry in training_data]
train_labels = [entry.get('response (string)', '') for entry in training_data]

# Remove empty patterns or labels
train_data = [(pattern, label) for pattern, label in zip(train_patterns, train_labels) if pattern and label]

# Check if train_data is empty after filtering
if not train_data:
    st.error("No valid training data found. Please check the training data.")
    st.stop()  # Stop execution if there's no valid training data

# Unzip into separate lists
train_patterns, train_labels = zip(*train_data)

# Convert to NumPy array for consistency
y_train = np.array(train_labels)

# Convert training data to DataFrame for easier manipulation
train_df = pd.DataFrame({
    'patterns': train_patterns,
    'labels': y_train
})

# Feature extraction
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['patterns'])

# Check shapes before fitting
st.write(f"X_train shape: {X_train.shape}, y_train length: {len(y_train)}")

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Load testing data from JSON file
with open(r'C:\Users\User\NLP\testing_data.json', 'r') as json_file:
    testing_data = json.load(json_file)

# Check if testing data is empty
if not testing_data:
    st.error("Testing data is empty. Please check the testing data source.")
    st.stop()  # Stop execution if there's no testing data

# Prepare testing data
test_patterns = [entry.get('instruction (string)', '') for entry in testing_data]
test_labels = [entry.get('response (string)', '') for entry in testing_data]

# Remove empty patterns or labels
test_data = [(pattern, label) for pattern, label in zip(test_patterns, test_labels) if pattern and label]

# Check if test_data is empty after filtering
if not test_data:
    st.error("No valid testing data found. Please check the testing data.")
    st.stop()  # Stop execution if there's no valid testing data

# Unzip into separate lists
test_patterns, test_labels = zip(*test_data)

# Convert to NumPy array for consistency
y_test = np.array(test_labels)

# Convert testing data to DataFrame for easier manipulation
test_df = pd.DataFrame({
    'patterns': test_patterns,
    'labels': y_test
})

# Feature extraction for testing data
X_test = vectorizer.transform(test_df['patterns'])

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display results
st.write(f"Model accuracy: {accuracy:.2f}")
