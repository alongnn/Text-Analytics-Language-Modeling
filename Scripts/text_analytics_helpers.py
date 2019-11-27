"""
Author: Saurabh Annadate

This Script contains all text analytics helper functions

"""

import nltk
import os
import multiprocessing
import re
import logging
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.sequence import pad_sequences

logger = logging.getLogger()

from nltk.tokenize import sent_tokenize, word_tokenize

def replace_occ(string, char):
    """Replaces multiple occurances of the given character with a single occurance in the given string
    """
    pattern = char + '{2,}'
    string = re.sub(pattern, char, string) 
    return string 

def corpus_to_sent_tokens(doc):
    """
    Converts the given cleaned corpus to a list of sentence tokens

    Args:
        doc: String containing the corpus

    Returns:
        List of List containing tokens. eg [["My", "name", "is", "John", "."]["I", "am", "a", ......]]
    """
    
    logger.debug("Running the corpus_to_sent_tokens function.")

    #Cobverting to sentences
    sent_tokens = sent_tokenize(doc)

    #Converting to word tokens
    token_list = []
    for sent in sent_tokens:
        token_list.append(word_tokenize(sent))

    return token_list

def split_input_target(chunk):
    """Splits a given sequence into prediction and label
    """
    input_text = chunk[:-1]
    target_text = chunk[-1]
    return input_text, target_text

def char_data_generator(text, batch_size, char2idx, seq_length, vocab):
    """Generator function to yield batches of data for training character level neural net

    Args:
        text (String): Data for creating training and eval sets
        batch_size (int): size of batch
        char2idx (Dict): Dictionary mapping character to index
        seq_length (int): Sequence length to generate per training example
        vocab (string): All characters present in the dataset

    Yields:
        X,y : Training / Evaluation data of desired batch size
    """

    itr = 0

    lb = MultiLabelBinarizer()
    lb.fit(vocab)
    
    while True:
        training_data = []
        
        # keep looping until we reach our batch size
        while len(training_data) < batch_size:
            if itr > (len(text) - seq_length):
                itr = 0
            training_data.append(text[itr:itr+seq_length+1])
            itr = itr + 1

        training_data = list(map(split_input_target, training_data))
        train_texts = [i[0] for i in training_data]
        
        train_texts_indices = []
        for k in train_texts:
            train_texts_indices.append([char2idx[c] for c in k])
        
        train_labels = [i[1] for i in training_data]

        Y = lb.transform(train_labels)
        Y = np.array(Y)

        x_data = pad_sequences(train_texts_indices, maxlen=int(seq_length))

        yield (x_data, Y)