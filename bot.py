import os
import json
import random
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load training data from JSON file
with open(r'C:\Users\User\NLP\training_data.json', 'r') as json_file:
    training_data = json.load(json_file)

# Prepare training data
train_patterns = []
train_tags = []

for entry in training_data:
    train_patterns.append(entry['instruction (string)'])  # Adjust key name if needed
    train_tags.append(entry['output (string)'])            # Adjust key name if needed

# Vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_patterns)  # Vectorize training instructions
y_train = train_tags                                  # Outputs for training

# Train the Model
clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(X_train, y_train)

# Function to handle chatbot responses
def chatbot(input_text):
    # Transform the input text into the same format as the training data
    input_vector = vectorizer.transform([input_text])

    # Predict the tag using the trained model
    predicted_tag = clf.predict(input_vector)[0]

    # Search for a matching response in the training data
    for entry in training_data:
        if entry['output (string)'] == predicted_tag:
            # Return a random response from matching entries
            return random.choice(entry['output (string)'])  # If multiple responses exist for a tag

    return "I'm sorry, I don't understand that question."  # Fallback response if no match is found

# Streamlit interface
def main():
    st.title("Menstrual Health Awareness Bot")
    st.write("Welcome to the chatbot!")

    user_input = st.text_input("You:", key="user_input")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key="chatbot_response")

if __name__ == '__main__':
    main()
