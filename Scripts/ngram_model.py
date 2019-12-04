"""
Author: Saurabh Annadate

Script to train a word level neural network.

"""

import logging
import os
import yaml
import logging
import datetime
import requests
import time
import string
import random
import numpy as np
import pickle

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

from Scripts.helpers import create_dir
from Scripts.text_analytics_helpers import corpus_to_sent_tokens

logger = logging.getLogger()

def ngram_model(args):
    """
    This functions trains and saves a n-gram model

    Args:
        None
    
    Returns:
        None
    """
    logger.debug("Running the ngram_model function")

    #Loading the config
    with open(os.path.join("Config","config.yml"), "r") as f:
        config = yaml.safe_load(f)

    #Creating folder for this run
    create_dir(os.path.join("Models", "ngram_models"))

    #Loading the document
    file = open(os.path.join(config["create_corpus"]["save_location"], "processed_data.txt"), 'r', encoding="UTF-8")
    text = file.read()
    file.close()

    #Tokenizing the document
    text_tokenized = corpus_to_sent_tokens(text)

    validation_split = config["n_gram"]["validation_split"]
    index_split = round(len(text_tokenized) * (1-validation_split))

    training_text = text_tokenized[0:index_split]
    val_text = text_tokenized[index_split+1:]

    val_text = [item for sublist in val_text for item in sublist]

    tstart = datetime.datetime.now()

    #Fitting the language model
    train, vocab = padded_everygram_pipeline(config["n_gram"]["gram_count"], training_text)
    lm = MLE(config["n_gram"]["gram_count"])
    lm.fit(train, vocab)

    train_time = datetime.datetime.now() - tstart

    with open(os.path.join("Models", "ngram_models", config["n_gram"]["model_name"]+".pkl"), 'wb') as f:
        pickle.dump(lm, f)

    logger.info("Training complete. Model Saved.")

    f = open(os.path.join("Models", "ngram_models", config["n_gram"]["model_name"] + "_summary.txt"),"w+")
    f.write('Date of run: {} \n'.format(str(datetime.datetime.now())))
    
    f.write('\n\n\nModel Parameters:\n')
    f.write('Model Name: {}\n'.format(config["n_gram"]["model_name"]))
    f.write('n-Grams: {}\n'.format(config["n_gram"]["gram_count"]))
  
    f.write('\n\n\nModel Performance Metrics:\n')
    f.write("Total Train time = {}".format(train_time))
    f.close()
    
    logger.info('Model Summary Written')

    return
