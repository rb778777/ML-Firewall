from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd

app = Flask(__name__)


class XSSDetector(nn.Module):
    def __init__(self, n_classes):
        super(XSSDetector, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = bert_output.last_hidden_state[:, 0, :]
        output = self.drop(output)
        return self.out(output)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = XSSDetector(n_classes=2)
model.load_state_dict(torch.load('xss_detection_model.pth', map_location=torch.device('cpu')))
model.eval()


def preprocess(sentence):
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    return encoding['input_ids'], encoding['attention_mask']


def predict(sentence):
    input_ids, attention_mask = preprocess(sentence)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    return preds.item()

@app.route('/analyze', methods=['POST', 'GET'])
def analyze():
    if request.method == 'POST':
        data = request.form.get('data')
    else:
        data = request.args.get('data')

    if data is None:
        return jsonify({"error": "No data provided"}), 400

    prediction = predict(data)

    if prediction == 1:
        return jsonify({"result": "XSS detected"}), 403
    else:
        return jsonify({"result": "No XSS detected"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

