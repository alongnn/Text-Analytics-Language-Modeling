"""
Author: Saurabh Annadate

Enables the command line execution of multiple modules within Scripts/

"""

import os
import argparse
import logging
import logging.config
import yaml

with open(os.path.join("Config","config.yml"), "r") as f:
    config = yaml.safe_load(f)

# The logging configurations are called from local.conf
logging.config.fileConfig(os.path.join("Config","logging_local.conf"))
logger = logging.getLogger(config['logging']['LOGGER_NAME'])

from Scripts.fetch_data import fetch_data
from Scripts.clean_data import clean_data
from Scripts.create_corpus import create_corpus
from Scripts.neural_char_level_model import char_level_neural_net
from Scripts.word_embedding_model import word_embedding_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run components of the run source code")
    subparsers = parser.add_subparsers()
    
    # Sub-parser for scraping the data
    sb_fetch = subparsers.add_parser("fetch_data", description="Fetch the raw data from the source")
    sb_fetch.set_defaults(func=fetch_data)

    # Sub-parser for running logistic regression
    sb_clean = subparsers.add_parser("clean_data", description="Runs a logistic regression on the data")
    sb_clean.set_defaults(func=clean_data)

    # Sub-parser for creating corpus
    corp_create = subparsers.add_parser("create_corpus", description="Create corpus to train models")
    corp_create.add_argument("--docCount", default=500, help="Count of books to be used for creating the corpus")
    corp_create.set_defaults(func=create_corpus)

    # Sub-parser for running word2vec model
    char_nn = subparsers.add_parser("run_w2v", description="Runs word2vec model on the data")
    char_nn.set_defaults(func=word_embedding_model)

    # Sub-parser for running character level neural net
    char_nn = subparsers.add_parser("run_char_nn", description="Runs a cnn on the data")
    char_nn.set_defaults(func=char_level_neural_net)

    # # Sub-parser for predicting using cnn
    # ft = subparsers.add_parser("predict_cnn", description="Predicts using cnn on the data")
    # ft.set_defaults(func=predict_cnn)

    # # Sub-parser for predicting using cnn
    # ft = subparsers.add_parser("predict_svm", description="Predicts using svm on the data")
    # ft.set_defaults(func=predict_svm)

    args = parser.parse_args()
    args.func(args)
