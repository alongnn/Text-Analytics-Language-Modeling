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
from keras.layers import Dense, LSTM, Embedding, Flatten, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

from Scripts.text_analytics_helpers import split_input_target
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
        text = text[0:config["gen_training"]["training_size"]]

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

    #Vectorizing the text
    text_as_int = np.array([char2idx[c] for c in text])

    #Creating training data
    # The maximum length sentence we want for a single input in characters
    seq_length = config["char_nn"]["seq_length"]

    logger.debug("Creating training data now.")
    training_data = []
    for itr in range(len(text_as_int) - seq_length):
        training_data.append(text_as_int[itr:itr+seq_length+1])

    training_data = list(map(split_input_target, training_data))

    train_texts_indices = [i[0] for i in training_data]
    train_labels = [i[1] for i in training_data]

    train_labels = keras.utils.to_categorical(train_labels)
    y_data = train_labels

    x_data = pad_sequences(train_texts_indices, maxlen=int(seq_length))

    #Defining model
    logger.debug("Training data and labels generated. Defining model now.")
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, config["char_nn"]["embedding_dim"],
                                input_length=seq_length,
                                ))

    model.add(LSTM(units = config["char_nn"]["rnn_units"], 
                    return_sequences=True, 
                    recurrent_initializer='glorot_uniform', 
                    dropout=0.3
                    ))

    model.add(LSTM(units = config["char_nn"]["rnn_units"], 
                    return_sequences=False, 
                    recurrent_initializer='glorot_uniform', 
                    dropout=0.3
                    ))

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

    early_stopping = EarlyStopping(patience=config["char_nn"]["patience"])
    Y = np.array(y_data)

    logger.debug("Fitting model now.")
    fit = model.fit(x_data,
                    Y,
                    batch_size=config["char_nn"]["BATCH_SIZE"],
                    epochs=config["char_nn"]["epochs"],
                    validation_split=config["char_nn"]["VALIDATION_SPLIT"],
                    verbose=1,
                    callbacks=[early_stopping])

    model.save(os.path.join("Models", config["char_nn"]["model_name"], config["char_nn"]["model_name"] + ".model"))
    logger.info("Final val_categorical_crossentropy = {}".format(fit.history['val_categorical_crossentropy']))
    logger.info("Training complete.")

    return