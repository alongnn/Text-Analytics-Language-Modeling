"""
Author: Saurabh Annadate

Script containing api for predicting using word_neural_model

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
from Scripts.text_analytics_helpers import get_embd

from gensim.models import Word2Vec, KeyedVectors
from Scripts.text_analytics_helpers import corpus_to_sent_tokens

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

#Loading the w2v model
w2vmodel = Word2Vec.load(config["word_api"]["w2v_model"])
embd_size = config["word_api"]["embd_size"]

model = load_model(os.path.join("Models", config["word_api"]["model_name"], config["word_api"]["model_name"] + ".model"))
graph = tf.get_default_graph()

#Clean text
def process_text(text):
    """Processes the input text and returns the input in model ready format
    """
    seq_length = config["word_api"]["seq_length"]
    text = text[-seq_length:]
    
    train_texts_embd = [get_embd(z, w2vmodel, embd_size) for z in text]
    
    x_data = np.array([train_texts_embd])
    return x_data


def get_next_word(model_pred):
    """Based on the model prediction, yield next word on the basis of the idx2char dictionary
    """
    pred_vect = np.array(model_pred[0])
    next_word = w2vmodel.wv.most_similar(positive = [pred_vect], topn=1)[0][0]
    return next_word

def get_prediction(text, length = 500):
    """Returns the predicted text for the given length
    """
    for i in range(int(length)):
        x_data = process_text(text)
        model_pred = model.predict(x_data)
        text = text + " " + get_next_word(model_pred)
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
        
        text_tokenized = corpus_to_sent_tokens(inp_text)
        text_tokenized = [item for sublist in text_tokenized for item in sublist]
        
        if len(text_tokenized) < config["word_api"]["seq_length"]:
            output = {"input_text": inp_text, "input_length": inp_len, "status": 400}
        else:
            pred_text = get_prediction(inp_text, inp_len)
            output = {"input_text": inp_text, "input_length": inp_len, "status": 200, "pred_text": pred_text}

    return output

def run_word_api(args):
    app.run(debug=False)