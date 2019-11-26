"""
Author: Saurabh Annadate

Script to concatenate the clean data anc create corpus

"""

import logging
import os
import yaml
import logging
import datetime
import requests
import time

logger = logging.getLogger()

from Scripts.text_analytics_helpers import replace_occ

def create_corpus(args):
    """
    Function to create the corpus. Creating corpus involves concatenating all the books and creating a single corpus to train the LMs on.

    Args:
        None

    Returns:
        None
    
    """
    logger.debug("Running the clean_data function")

    #Loading the config
    with open(os.path.join("Config","config.yml"), "r") as f:
        config = yaml.safe_load(f)

    file_list = [i for i in os.listdir(config["clean_data"]["save_location"]) if i[-3:]=="txt"]

    concat_contents = []
 
    for i in file_list[:int(args.docCount)]:

        f = open(os.path.join(config["clean_data"]["save_location"], i), "r", encoding="UTF-8")
        contents = f.readlines()
        
        concat_contents.append(''.join(contents))
        f.close()

    #Concatinating the list
    write_string = ''.join(concat_contents)

    #Cleaning the file
    write_string = replace_occ(write_string, '\n')
    write_string = write_string.split('\n')
    write_string = ' '.join(write_string)
    write_string = replace_occ(write_string, " ")

    f = open(os.path.join(config["create_corpus"]["save_location"], "processed_data.txt"),"w+", encoding="UTF-8")
    f.write(write_string)
    f.close()

    logger.info("Processed file generated")

    return