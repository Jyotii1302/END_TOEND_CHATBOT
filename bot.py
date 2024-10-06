import json
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

# Load training data from the JSON file using requests
url = 'https://raw.githubusercontent.com/Jyotii1302/END_TOEND_CHATBOT/main/training_data.json'
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    training_data = response.json()
else:
    raise ValueError(f"Error fetching data: {response.status_code}")

# Check the structure of the loaded data
if not training_data or len(training_data) == 0:
    raise ValueError("No valid training data found. Please check the training data.")

# Prepare training data
train_patterns = [entry['instruction (string)'] for entry in training_data]
train_labels = [entry['output (string)'] for entry in training_data]

# Check for unique labels
unique_labels = set(train_labels)
if len(unique_labels) < 2:
    raise ValueError("This model requires at least two classes for training. Found only: " + str(unique_labels))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_patterns, train_labels, test_size=0.2, random_state=42)

# Create a pipeline for vectorization and model fitting
model = make_pipeline(CountVectorizer(), LogisticRegression())

# Fit the model
model.fit(X_train, y_train)

# Now you can use model.predict to make predictions
print("Model trained successfully!")
