from flask import Flask, request, abort
import joblib
import pandas as pd
import urllib.parse
import html

app = Flask(__name__)


model = joblib.load('isolation_forest_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


file_path = 'xssdata.xlsx'
df = pd.read_excel(file_path)
malicious_payloads = df[df['Label'] == 1]['Payload'].tolist()  

def decode_payload(payload):
    
    decoded_payload = urllib.parse.unquote(payload)
    
    decoded_payload = html.unescape(decoded_payload)
    
    while decoded_payload != payload:
        payload = decoded_payload
        decoded_payload = urllib.parse.unquote(payload)
        decoded_payload = html.unescape(decoded_payload)
    return decoded_payload

@app.before_request
def block_xss():
    payload = request.args.get('payload') or request.form.get('payload')
    if payload:
        decoded_payload = decode_payload(payload)
        print(f"Received payload: {payload}")
        print(f"Decoded payload: {decoded_payload}")

        
        if decoded_payload in malicious_payloads:
            print(f"Blocked payload (exact match): {decoded_payload}")
            abort(403)  

        
        vectorized_payload = vectorizer.transform([decoded_payload])
        result = model.decision_function(vectorized_payload)
        print(f"Decision function result: {result[0]}")

        
        if result[0] < 0.05:  
            print(f"Blocked payload (ML): {decoded_payload}")
            abort(403)  

@app.route('/', methods=['GET', 'POST'])
def index():
    return "Request is allowed."

if __name__ == '__main__':
    app.run(debug=True)

