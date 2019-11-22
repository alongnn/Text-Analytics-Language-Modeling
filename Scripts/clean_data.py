"""
Author: Saurabh Annadate

Script to clean the raw data

"""

import logging
import os
import yaml
import logging
import datetime
import requests
import time

logger = logging.getLogger()

def clean_data(args):
    """
    Function to clean the data. Cleaning involves removing the beginning, ending, and selecting only english books.
    All documents will be stored in location specified in config["clean_data"]["save_location"]

    Args:
        None

    Returns:
        None

    """
    logger.debug("Running the clean_data function")

    #Loading the config
    with open(os.path.join("Config","config.yml"), "r") as f:
        config = yaml.safe_load(f)

    file_list = [i for i in os.listdir(config["fetch_data"]["save_location"]) if i[-3:]=="txt"]

    for i in file_list:

        f = open(os.path.join(config["fetch_data"]["save_location"], i), "r", encoding="UTF-8")
        contents = f.readlines()
        
        #Filtering for english books and then removing the starting and ending sections
        if "Language: English\n" in contents:
            
            filter_contents = [i[:39] for i in contents]

            start_phrase = "*** START OF THIS PROJECT GUTENBERG EBO"
            end_phrase = "*** END OF THIS PROJECT GUTENBERG EBOOK"

            try:
                start_index = filter_contents.index(start_phrase)
            except Exception as e:
                logger.error(e)
                start_index = 0

            try:
                end_index = filter_contents.index(end_phrase)
            except Exception as e:
                logger.error(e)
                end_index = len(filter_contents)

            final_contents = contents[start_index + 1: end_index]

            #Writing the new file
            write_string = ''.join(final_contents)

            f = open(os.path.join(config["clean_data"]["save_location"], i),"w+", encoding="UTF-8")
            f.write(write_string)
            f.close()

            logger.debug("File {} Cleaned.".format(i))

    return