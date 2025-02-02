import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("Phishing_Email.csv")  # Ensure this file is inside the backend folder

# Check column names
print(df.columns)

# Assuming 'email_text' contains the text and 'label' contains 0 (safe) or 1 (phishing)
X = df["email_text"]
y = df["label"]

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save Model and Vectorizer
joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
