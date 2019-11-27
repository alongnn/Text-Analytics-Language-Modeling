"""
Author: Saurabh Annadate

Script to train a character level neural network.

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
from keras.callbacks import EarlyStopping

from Scripts.text_analytics_helpers import split_input_target, char_data_generator
from Scripts.helpers import create_dir

logger = logging.getLogger()

def char_level_neural_net(args):
    """
    This functions trains and saves a character level neural network

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
    create_dir(os.path.join("Models", config["char_nn"]["model_name"]))

    #Loading the document
    file = open(os.path.join(config["create_corpus"]["save_location"], "processed_data.txt"), 'r', encoding="UTF-8")
    text = file.read()
    file.close()

    logger.debug("Total characters in the corpus : {}".format(len(text)))

    #Limiting training size based on config:
    if config["gen_training"]["char_nn_training_size"] != -1:
        text = text[0:config["gen_training"]["char_nn_training_size"]]

    logger.debug("After limiting training size, total characters in the corpus : {}".format(len(text)))

    #Generating vocabulary of the characters
    vocab = sorted(set(text))

    logger.debug("Total unique characters in the corpus : {}".format(len(vocab)))

    # Creating a mapping from unique characters to indices
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = {i:u for i, u in enumerate(vocab)}

    #Saving dictionaries
    with open(os.path.join("Models", config["char_nn"]["model_name"], config["char_nn"]["model_name"] + "_char2idx.pickle"), 'wb') as handle:
        pickle.dump(char2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join("Models", config["char_nn"]["model_name"], config["char_nn"]["model_name"] + "_idx2char.pickle"), 'wb') as handle:
        pickle.dump(idx2char, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.debug("Dictionaries created and saved.")

    #Creating training and validation split
    validation_split = config["char_nn"]["validation_split"]
    index_split = round(len(text) * (1-validation_split))
    training_text = text[0:index_split]
    val_text = text[index_split+1:]

    batch_size = config["char_nn"]["batch_size"]
    seq_length = config["char_nn"]["seq_length"]

    #Defining training and validation data generators
    train_gen = char_data_generator(training_text, batch_size, char2idx, seq_length, vocab)
    val_gen = char_data_generator(val_text, batch_size, char2idx, seq_length, vocab)

    #Defining model
    logger.debug("Training data and labels generated. Defining model now.")
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, config["char_nn"]["embedding_dim"],
                                input_length=seq_length,
                                ))

    if config["char_nn"]["rnn_type"] == "lstm":
        if config["char_nn"]["rnn_layers"] > 1:
            for _ in range(config["char_nn"]["rnn_layers"] - 1):
                model.add(LSTM(units = config["char_nn"]["rnn_units"], 
                                return_sequences=True, 
                                recurrent_initializer='glorot_uniform', 
                                dropout=config["char_nn"]["dropout"]
                                ))

        model.add(LSTM(units = config["char_nn"]["rnn_units"], 
                        return_sequences=False, 
                        recurrent_initializer='glorot_uniform', 
                        dropout=config["char_nn"]["dropout"]
                        ))
    
    elif config["char_nn"]["rnn_type"] == "gru":
        if config["char_nn"]["rnn_layers"] > 1:
            for _ in range(config["char_nn"]["rnn_layers"] - 1):
                model.add(GRU(units = config["char_nn"]["rnn_units"], 
                                return_sequences=True, 
                                recurrent_initializer='glorot_uniform', 
                                dropout=config["char_nn"]["dropout"]
                                ))

        model.add(GRU(units = config["char_nn"]["rnn_units"], 
                        return_sequences=False, 
                        recurrent_initializer='glorot_uniform', 
                        dropout=config["char_nn"]["dropout"]
                        ))

    else:
        logger.error("rnn_type should be either 'lstm' or 'gru'.")
        return

    model.add(Dense(len(vocab), 
                        activation='softmax',
                        kernel_regularizer=l2(config["char_nn"]["l2_penalty"]),
                        bias_regularizer=l2(config["char_nn"]["l2_penalty"]),
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros'
                        ))

    print(model.summary())
    
    logger.debug("Compiling Model now.")
    model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy', 'categorical_crossentropy'])
    
    logger.debug("Fitting model now.")
    
    tstart = datetime.datetime.now()
    
    fit = model.fit_generator(train_gen,
                    steps_per_epoch=(len(training_text) - seq_length)// batch_size,
                    validation_data=val_gen,
                    validation_steps=(len(val_text)  - seq_length)// batch_size,
                    epochs=config["char_nn"]["epochs"],
                    verbose=1)

    train_time = datetime.datetime.now() - tstart

    model.save(os.path.join("Models", config["char_nn"]["model_name"], config["char_nn"]["model_name"] + ".model"))
    logger.info("Final val_categorical_crossentropy = {}".format(fit.history['val_categorical_crossentropy'][-1]))
    logger.info("Training complete. Writing summary and performance file.")

    f = open(os.path.join("Models", config["char_nn"]["model_name"], config["char_nn"]["model_name"] + "_summary.txt"),"w+")
    f.write('Date of run: {} \n'.format(str(datetime.datetime.now())))
    f.write('Model Summary:\n')
    model.summary(print_fn=lambda x: f.write(x + '\n'))

    f.write('\n\n\nModel Parameters:\n')
    f.write('Model Name: {}\n'.format(config["char_nn"]["model_name"]))
    f.write('Train Data Character length: {}\n'.format(config["gen_training"]["char_nn_training_size"]))
    f.write('Sequence Length: {}\n'.format(config["char_nn"]["seq_length"]))
    f.write('Batch Size: {}\n'.format(config["char_nn"]["batch_size"]))
    f.write('Embedding Dimensions: {}\n'.format(config["char_nn"]["embedding_dim"]))
    f.write('RNN Units: {}\n'.format(config["char_nn"]["rnn_units"]))
    f.write('Epochs: {}\n'.format(config["char_nn"]["epochs"]))
    f.write('Validation Split: {}\n'.format(config["char_nn"]["validation_split"]))
    f.write('L2 penalty: {}\n'.format(config["char_nn"]["l2_penalty"]))

    f.write('\n\n\nModel Performance Metrics:\n')
    f.write("val_categorical_crossentropy = {}".format(fit.history['val_categorical_crossentropy']))
    f.write("Total Train time = {}".format(train_time))
    f.close()
    
    logger.info('Model Summary Written')

    return
