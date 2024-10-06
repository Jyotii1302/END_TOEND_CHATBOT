import os
import json
import random
import ssl
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set up SSL context to bypass verification (use with caution)
ssl._create_default_https_context = ssl._create_unverified_context

# Set NLTK data path and download necessary resources
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt', quiet=True)  # Download punkt without printing output

# Load intents from a JSON file
with open(r'C:\Users\User\NLP\intents.json', 'r') as file:
    intents = json.load(file)

# Prepare tags and patterns for training
tags = []
patterns = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Create vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Transform patterns into TF-IDF features and train the classifier
X = vectorizer.fit_transform(patterns)
y = tags
clf.fit(X, y)

# Chatbot response function
def chatbot(input_text):
    input_vector = vectorizer.transform([input_text])
    predicted_tag = clf.predict(input_vector)[0]
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:  # Correctly compare the tag
            return random.choice(intent['responses'])  # Randomly select a response

# Main Streamlit application
def main():
    import nltk  # Importing here to see if it resolves the issue
    import ssl
    
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.data.path.append(os.path.abspath("nltk_data"))
    nltk.download('punkt', quiet=True)  # Download punkt silently

    st.title("Menstrual Health Awareness Chatbot")
    st.write("Welcome to the chatbot!")

    user_input = st.text_input("You:", key="user_input")

    if user_input:
        response = chatbot(user_input)
        st.write(response)  # Display the chatbot's response
