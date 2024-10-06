import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load training data from GitHub repository
with open('https://raw.githubusercontent.com/jyotii1302/end_toend_chatbot/main/training_data.json', 'r') as json_file:
    training_data = json.load(json_file)

# Load testing data from GitHub repository
with open('https://raw.githubusercontent.com/jyotii1302/end_toend_chatbot/main/testing_data.json', 'r') as json_file:
    testing_data = json.load(json_file)

# Prepare training data
train_patterns = [entry['instruction (string)'] for entry in training_data]
train_labels = [entry['response (string)'] for entry in training_data]

# Convert training data to DataFrame for easier manipulation
train_df = pd.DataFrame({
    'patterns': train_patterns,
    'labels': train_labels
})

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_df['patterns'], train_df['labels'], test_size=0.2, random_state=42)

# Train the model (example using Logistic Regression)
model = LogisticRegression()
model.fit(X_train.values.reshape(-1, 1), y_train)  # Reshape if necessary

# Validate the model
predictions = model.predict(X_val.values.reshape(-1, 1))
accuracy = accuracy_score(y_val, predictions)

print(f"Validation Accuracy: {accuracy * 100:.2f}%")
