"""
Author: Saurabh Annadate

Script containing api for predicting using ngram model

"""

from flask import Flask, request, jsonify

import os
import re
import nltk
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
import random

logger = logging.getLogger()

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

from Scripts.text_analytics_helpers import corpus_to_sent_tokens

logger = logging.getLogger()

app = Flask(__name__)

#Loading the config
with open(os.path.join("Config","config.yml"), "r") as f:
    config = yaml.safe_load(f)

#Loading the ngram model
with open(os.path.join("Models", "ngram_models", config["ngram_api"]["model_name"]+".pkl"), 'rb') as f:
    lm = pickle.load(f)

def get_prediction(text, length = 500):
    """Returns the predicted text for the given length
    """
    gen_text = lm.generate(int(length), text_seed=text, random_seed=12345)
    return gen_text

# request model prediction
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        inp_text = request.args.get('query')
        inp_len = request.args.get('length')
    if request.method == 'POST':
        inp_text = request.form.get('query')
        inp_len = request.args.get('length')
    
    
    text_tokenized = corpus_to_sent_tokens(inp_text)
    text_tokenized = [item for sublist in text_tokenized for item in sublist]
    
    pred_text = get_prediction(text_tokenized, inp_len)
    out_text = inp_text
    for k in pred_text:
        out_text = out_text + " " + k
    
    out_text = out_text.replace("<s>", "").replace("</s>",".")
    output = {"input_text": inp_text, "input_length": inp_len, "status": 200, "pred_text": out_text}

    return output

def run_ngram_api(args):
    app.run(debug=False)