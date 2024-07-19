from flask import Flask, request, abort
import joblib
import pandas as pd
import urllib.parse
import html

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('isolation_forest_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to decode the payload
def decode_payload(payload):
    decoded_payload = urllib.parse.unquote(payload)
    decoded_payload = html.unescape(decoded_payload)
    
    # Ensure full decoding of the payload
    while decoded_payload != payload:
        payload = decoded_payload
        decoded_payload = urllib.parse.unquote(payload)
        decoded_payload = html.unescape(decoded_payload)
    return decoded_payload

# Middleware to block XSS payloads
@app.before_request
def block_xss():
    payload = request.args.get('payload') or request.form.get('payload')
    if payload:
        decoded_payload = decode_payload(payload)
        print(f"Received payload: {payload}")
        print(f"Decoded payload: {decoded_payload}")  

        # Vectorize the decoded payload
        vectorized_payload = vectorizer.transform([decoded_payload])
        result = model.decision_function(vectorized_payload)
        print(f"Decision function result: {result[0]}")

        # Block if the decision function result is below the threshold
        if result[0] < 0.05:  
            print(f"Blocked payload (ML): {decoded_payload}")
            abort(403)  # Forbidden

# Define the main route
@app.route('/', methods=['GET', 'POST'])
def index():
    return "Request is allowed."

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

