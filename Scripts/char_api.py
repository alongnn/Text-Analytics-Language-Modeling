"""
Author: Saurabh Annadate

Script containing api for predicting using char_neural_model

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

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.backend import set_session

app = Flask(__name__)

cnfg = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
cnfg.gpu_options.allow_growth = True
session = tf.Session(config=cnfg)
set_session(session)

#Loading the config
with open(os.path.join("Config","config.yml"), "r") as f:
    config = yaml.safe_load(f)

#Loading the dictionaries
with open(os.path.join("Models", config["char_api"]["model_name"], config["char_api"]["model_name"] + "_char2idx.pickle"), 'rb') as handle:
    char2idx = pickle.load(handle)

with open(os.path.join("Models", config["char_api"]["model_name"], config["char_api"]["model_name"] + "_idx2char.pickle"), 'rb') as handle:
    idx2char = pickle.load(handle)

model = load_model(os.path.join("Models", config["char_api"]["model_name"], config["char_api"]["model_name"] + ".model"))
graph = tf.get_default_graph()

#Clean text
def process_text(text):
    """Processes the input text and returns the input in model ready format
    """
    seq_length = config["char_api"]["seq_length"]
    text = text[-seq_length:]
    text_indices = []

    text_indices = [char2idx[c] for c in text]

    x_data = np.array([text_indices])
    return x_data

def get_next_char(model_pred):
    """Based on the model prediction, yield next character on the basis of the idx2char dictionary
    """
    model_pred_l = list(model_pred)
    max_pred = max(model_pred_l[0])
    max_index = list(model_pred_l[0]).index(max_pred)
    next_char = idx2char[max_index]
    return next_char

def get_prediction(text, length = 500):
    """Returns the predicted text for the given length
    """
    for i in range(int(length)):
        x_data = process_text(text)
        model_pred = model.predict(x_data)
        text = text + str(get_next_char(model_pred))
        logger.debug("At Index {}.".format(i))
    return text

# request model prediction
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        inp_text = request.args.get('query')
        inp_len = request.args.get('length')
    if request.method == 'POST':
        inp_text = request.form.get('query')
        inp_len = request.args.get('length')
    
    with graph.as_default():
        if len(inp_text) < config["char_api"]["seq_length"]:
            output = {"input_text": inp_text, "input_length": inp_len, "status": 400}
        else:
            pred_text = get_prediction(inp_text, inp_len)
            output = {"input_text": inp_text, "input_length": inp_len, "status": 200, "pred_text": pred_text}

    return output

def run_char_api(args):
    app.run(debug=False)