import pandas as pd
import csv
import json
import os
import random
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

# Set SSL context for NLTK data download
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load training data from CSV
def load_training_data(csv_file_path):
    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        training_data = [row for row in csv_reader]
    return training_data

# Load testing data from CSV
def load_testing_data(csv_file_path):
    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        testing_data = [row for row in csv_reader]
    return testing_data

# Convert CSV files to JSON files (if needed)
def save_to_json(data, json_file_path):
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Load training and testing data
training_data = load_training_data(r'C:\Users\User\Desktop\Datasets\Menstrual health data\Training_data.csv')
testing_data = load_testing_data(r'C:\Users\User\Desktop\Datasets\Menstrual health data\Testing_data.csv')

# Prepare training patterns and tags
train_patterns = [entry['instruction (string)'] for entry in training_data]
train_tags = [entry['output (string)'] for entry in training_data]

# Prepare testing patterns and tags
test_patterns = [entry['instruction (string)'] for entry in testing_data]
test_tags = [entry['output (string)'] for entry in testing_data]

# Vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_patterns)  # Vectorize training instructions
y_train = train_tags                                 # Outputs for training

X_test = vectorizer.transform(test_patterns)        # Vectorize testing instructions
y_test = test_tags                                   # Outputs for testing

# Train the Model
clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(X_train, y_train)

# Evaluate the Model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Chatbot Functionality
def chatbot(input_text):
    input_vector = vectorizer.transform([input_text])  # Transform the input text
    predicted_tag = clf.predict(input_vector)[0]       # Predict the tag

    # Collect responses based on the predicted tag
    responses = [entry['output (string)'] for entry in training_data if entry['output (string)'] == predicted_tag]

    if responses:
        return random.choice(responses)  # Return a random response if found
    return "I'm sorry, I don't understand that question."  # Fallback response

# Streamlit Application
def main():
    st.title("Menstrual Health Awareness Bot")
    st.write("Welcome to the chatbot! Feel free to ask questions about menstrual health.")
    
    user_input = st.text_input("You:")
    if user_input:
        response = chatbot(user_input)  # Get response from the chatbot
        st.text_area("Chatbot:", value=response, height=100)  # Display the response

if __name__ == '__main__':
    main()
