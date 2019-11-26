"""
Author: Saurabh Annadate

This Script contains all text analytics helper functions

"""

import nltk
import os
import multiprocessing
import re
import logging

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

a = [[5,3,4,5,6,7], [1,2,3,4,5,6], [9,6,5,4,3,2]]

b = list(map(split_input_target, a))



