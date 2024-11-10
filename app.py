# -*- coding: utf-8 -*-
"""
Created on Thu Feb 1 17:14:52 2024

@author: shadw
"""
import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
from flask_cors import CORS

# Disable GPU access to avoid CUDA-related issues on Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)
CORS(app, resources={'/*': {'origins': '*'}})

# Define the model input class
class ModelInput:
    def __init__(self, content):
        self.content = content

# Load model and tokenizer
try:
    loaded_model = load_model('your_model.h5')  # Ensure 'your_model.h5' is in the project directory on Render
    loaded_tokenizer = pickle.load(open('tokenizer.sav', 'rb'))  # Ensure 'tokenizer.sav' is in the project directory on Render
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

@app.route('/fake_news_prediction', methods=['POST'])
def news_pred():
    try:
        # Get JSON input data
        input_data = request.get_json()
        input_parameters = ModelInput(**input_data)
        
        # Process input text
        text = [input_parameters.content]
        text = loaded_tokenizer.texts_to_sequences(text)
        text = pad_sequences(text, maxlen=1000)
        
        # Predict with the model
        prediction = loaded_model.predict(text)
        
        # Interpret the prediction result
        result = 'Fake' if prediction[0, 0] >= 0.5 else 'Real'
        
        return jsonify({'prediction': float(prediction[0, 0]), 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/")
def home():
    return "shadwar"

if __name__ == "__main__":
    app.run(debug=False)
