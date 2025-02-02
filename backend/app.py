from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib

app = Flask(__name__)

# Enable CORS
CORS(app)  # This will allow all domains to access the API

# Load Model & Vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    email_text = data.get("email_text", "")

    if not email_text:
        return jsonify({"error": "No email text provided"}), 400

    # Transform input
    email_tfidf = vectorizer.transform([email_text])
    prediction = model.predict(email_tfidf)[0]

    return jsonify({"prediction": "phishing" if prediction == 1 else "safe"})

if __name__ == "__main__":
    app.run(debug=True)

