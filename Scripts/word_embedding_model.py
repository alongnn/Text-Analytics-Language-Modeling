"""
Author: Saurabh Annadate

Script to train a word2vec model

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

from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec, KeyedVectors

from Scripts.helpers import create_dir
from Scripts.text_analytics_helpers import corpus_to_sent_tokens

logger = logging.getLogger()

def word_embedding_model(args):
    """
    This functions trains and saves a word2vec model of the given specification

    Args:
        None
    
    Returns:
        None
    """
    logger.debug("Running the word_embedding_model function")

    #Loading the config
    with open(os.path.join("Config","config.yml"), "r") as f:
        config = yaml.safe_load(f)

    #Creating folder for this run
    create_dir(os.path.join("Models", "word2vec"))

    tstart = datetime.datetime.now()

    #Loading the document
    file = open(os.path.join(config["create_corpus"]["save_location"], "processed_data.txt"), 'r', encoding="UTF-8")
    text = file.read()
    file.close()

    #Tokenize the corpus
    logger.debug("Tokenizing the text.")
    text_tokenized = corpus_to_sent_tokens(text)

    logger.debug("Text tokenized. Removing sentences with less than 3 words.")
    text_tokenized = [i for i in text_tokenized if len(i)>=3]

    logger.debug("Training data ready - Fitting word2vec model now.")
    path = get_tmpfile("word2vec.model")
    model = Word2Vec(text_tokenized, min_count=config["w2v_model"]["min_count"], size=config["w2v_model"]["size"], workers=config["w2v_model"]["workers"], window =config["w2v_model"]["window"], sg=config["w2v_model"]["sg"], hs=config["w2v_model"]["hs"])
    model.save(os.path.join("Models", "word2vec", config["w2v_model"]["model_name"] + ".model"))

    logger.info("word2vec model created in time {}".format(datetime.datetime.now() - tstart))

    return