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
from Scripts.word_embedding_model import word_embedding_model
from Scripts.neural_char_level_model import char_level_neural_net
from Scripts.neural_word_level_model import word_level_neural_net
from Scripts.ngram_model import ngram_model
from Scripts.char_api import run_char_api
from Scripts.word_api import run_word_api
from Scripts.ngram_api import run_ngram_api

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
    w2v = subparsers.add_parser("run_w2v", description="Runs word2vec model on the data")
    w2v.set_defaults(func=word_embedding_model)

    # Sub-parser for running character level neural net
    char_nn = subparsers.add_parser("run_char_nn", description="Runs a character level nn on the data")
    char_nn.set_defaults(func=char_level_neural_net)

    # Sub-parser for running word level neural net
    word_nn = subparsers.add_parser("run_word_nn", description="Runs a word level nn on the data")
    word_nn.set_defaults(func=word_level_neural_net)

    # Sub-parser for running a ngram model
    ngram_mdl = subparsers.add_parser("run_ngram", description="Runs a ngram on the data")
    ngram_mdl.set_defaults(func=ngram_model)

    # Sub-parser for running the character API
    char_api = subparsers.add_parser("run_char_api", description="Runs the API for char neural net")
    char_api.set_defaults(func=run_char_api)

    # Sub-parser for running the word API
    word_api = subparsers.add_parser("run_word_api", description="Runs the API for word neural net")
    word_api.set_defaults(func=run_word_api)

    # Sub-parser for running the word API
    word_api = subparsers.add_parser("run_ngram_api", description="Runs the API for word neural net")
    word_api.set_defaults(func=run_ngram_api)

    args = parser.parse_args()
    args.func(args)