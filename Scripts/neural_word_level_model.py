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

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, Flatten, Dropout, GRU
from keras.regularizers import l2

from Scripts.helpers import create_dir
from Scripts.text_analytics_helpers import word_data_generator, corpus_to_sent_tokens

from gensim.models import Word2Vec, KeyedVectors

logger = logging.getLogger()

def word_level_neural_net(args):
    """
    This functions trains and saves a word level neural network

    Args:
        None
    
    Returns:
        None
    """
    logger.debug("Running the char_level_neural_net function")

    #Loading the config
    with open(os.path.join("Config","config.yml"), "r") as f:
        config = yaml.safe_load(f)

    #Creating folder for this run
    create_dir(os.path.join("Models", config["word_nn"]["model_name"]))

    w2vmodel = Word2Vec.load(config["word_nn"]["w2v_model"])

    #Loading the document
    file = open(os.path.join(config["create_corpus"]["save_location"], "processed_data.txt"), 'r', encoding="UTF-8")
    text = file.read()
    file.close()

    #Tokenizing the document
    text_tokenized = corpus_to_sent_tokens(text)
    text_tokenized = [item for sublist in text_tokenized for item in sublist]

    logger.debug("Total words in the corpus : {}".format(len(text_tokenized)))

    #Limiting training size based on config:
    if config["gen_training"]["word_nn_training_size"] != -1:
        text_tokenized = text_tokenized[0:config["gen_training"]["word_nn_training_size"]]

    logger.debug("After limiting training size, total words in the corpus : {}".format(len(text_tokenized)))

    #Creating training and validation split
    validation_split = config["word_nn"]["validation_split"]
    index_split = round(len(text_tokenized) * (1-validation_split))

    training_text = text_tokenized[0:index_split]
    val_text = text_tokenized[index_split+1:]
    
    
    logger.debug("Validation size: {}".format(len(training_text)))
    logger.debug("Eval size: {}".format(len(val_text)))
    
    
    batch_size = config["word_nn"]["batch_size"]
    seq_length = config["word_nn"]["seq_length"]
    embd_size = config["word_nn"]["embedding_dim"]

    #Defining training and validation data generators
    train_gen = word_data_generator(training_text, batch_size, w2vmodel, embd_size, seq_length)
    val_gen = word_data_generator(val_text, batch_size, w2vmodel, embd_size, seq_length)


    #Defining model
    logger.debug("Training data and labels generated. Defining model now.")
    model = Sequential()

    if config["word_nn"]["rnn_type"] == "lstm":
        if config["word_nn"]["rnn_layers"] > 1:
            for _ in range(config["word_nn"]["rnn_layers"] - 1):
                model.add(LSTM(units = config["word_nn"]["rnn_units"], 
                                return_sequences=True, 
                                recurrent_initializer='glorot_uniform', 
                                dropout=config["word_nn"]["dropout"]
                                ))

        model.add(LSTM(units = config["word_nn"]["rnn_units"], 
                        return_sequences=False, 
                        recurrent_initializer='glorot_uniform', 
                        dropout=config["word_nn"]["dropout"]
                        ))

    elif config["word_nn"]["rnn_type"] == "gru":
        if config["word_nn"]["rnn_layers"] > 1:
            for _ in range(config["word_nn"]["rnn_layers"] - 1):
                model.add(GRU(units = config["word_nn"]["rnn_units"], 
                                return_sequences=True, 
                                recurrent_initializer='glorot_uniform', 
                                dropout=config["word_nn"]["dropout"]
                                ))

        model.add(GRU(units = config["word_nn"]["rnn_units"], 
                        return_sequences=False, 
                        recurrent_initializer='glorot_uniform', 
                        dropout=config["word_nn"]["dropout"]
                        ))

    else:
        logger.error("rnn_type should be either 'lstm' or 'gru'.")
        return

    model.add(Dense(embd_size, 
                        activation='tanh',
                        kernel_regularizer=l2(config["word_nn"]["l2_penalty"]),
                        bias_regularizer=l2(config["word_nn"]["l2_penalty"]),
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros'
                        ))

    model.build((None, seq_length, embd_size))
    print(model.summary())
    
    logger.debug("Compiling Model now.")
    model.compile(loss='mean_squared_error',
                    optimizer='rmsprop',
                    metrics=['mse'])
    
    logger.debug("Fitting model now.")
    
    tstart = datetime.datetime.now()
    
    fit = model.fit_generator(train_gen,
                    steps_per_epoch=(len(training_text) - seq_length)// batch_size,
                    validation_data=val_gen,
                    validation_steps=(len(val_text)  - seq_length)// batch_size,
                    epochs=config["word_nn"]["epochs"],
                    verbose=1)

    train_time = datetime.datetime.now() - tstart

    model.save(os.path.join("Models", config["word_nn"]["model_name"], config["word_nn"]["model_name"] + ".model"))
    logger.info("Final MSE = {}".format(fit.history['val_mean_squared_error'][-1]))
    logger.info("Training complete. Writing summary and performance file.")

    f = open(os.path.join("Models", config["word_nn"]["model_name"], config["word_nn"]["model_name"] + "_summary.txt"),"w+")
    f.write('Date of run: {} \n'.format(str(datetime.datetime.now())))
    f.write('Model Summary:\n')
    model.summary(print_fn=lambda x: f.write(x + '\n'))

    f.write('\n\n\nModel Parameters:\n')
    f.write('Model Name: {}\n'.format(config["word_nn"]["model_name"]))
    f.write('Train Data Word length: {}\n'.format(config["gen_training"]["word_nn_training_size"]))
    f.write('Sequence Length: {}\n'.format(config["word_nn"]["seq_length"]))
    f.write('Batch Size: {}\n'.format(config["word_nn"]["batch_size"]))
    f.write('Embedding Dimensions: {}\n'.format(config["word_nn"]["embedding_dim"]))
    f.write('RNN Units: {}\n'.format(config["word_nn"]["rnn_units"]))
    f.write('Epochs: {}\n'.format(config["word_nn"]["epochs"]))
    f.write('Validation Split: {}\n'.format(config["word_nn"]["validation_split"]))
    f.write('L2 penalty: {}\n'.format(config["word_nn"]["l2_penalty"]))
    f.write('word2vev model: {}\n'.format(config["word_nn"]["w2v_model"]))

    f.write('\n\n\nModel Performance Metrics:\n')
    f.write("Final val_mse = {}\n".format(fit.history['val_mean_squared_error'][-1]))
    f.write("Total Train time = {}".format(train_time))
    f.close()
    
    logger.info('Model Summary Written')

    return
