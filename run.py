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
logging.config.fileConfig(os.path.join("config","logging_local.conf"))
logger = logging.getLogger(config['logging']['LOGGER_NAME'])

from Scripts.fetch_data import fetch_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run components of the run source code")
    subparsers = parser.add_subparsers()
    
    # Sub-parser for running data stats
    sb_fetch = subparsers.add_parser("fetch_data", description="Fetch the raw data from the source")
    sb_fetch.set_defaults(func=fetch_data)

    # # Sub-parser for running logistic regression
    # logit_reg = subparsers.add_parser("run_logit", description="Runs a logistic regression on the data")
    # logit_reg.set_defaults(func=fit_logistic_regression)

    # # Sub-parser for running svm
    # svm = subparsers.add_parser("run_svm", description="Runs a SVM on the data")
    # svm.set_defaults(func=fit_svm)

    # # # Sub-parser for running fasttext
    # # ft = subparsers.add_parser("run_fasttext", description="Runs a fasttext on the data")
    # # ft.set_defaults(func=fit_fasttext)

    # # Sub-parser for running cnn
    # ft = subparsers.add_parser("run_cnn", description="Runs a cnn on the data")
    # ft.set_defaults(func=fit_cnn)

    # # Sub-parser for predicting using cnn
    # ft = subparsers.add_parser("predict_cnn", description="Predicts using cnn on the data")
    # ft.set_defaults(func=predict_cnn)

    # # Sub-parser for predicting using cnn
    # ft = subparsers.add_parser("predict_svm", description="Predicts using svm on the data")
    # ft.set_defaults(func=predict_svm)

    args = parser.parse_args()
    args.func(args)
