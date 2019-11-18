"""
Author: Saurabh Annadate

Script to fetch the data from the website

"""

import logging
import os
import yaml
import logging
import datetime
import requests

logger = logging.getLogger()

def fetch_data(args):
    """
    Function to scrape and retreive the data from the website
    All documents will be stored in location specified in config["fetch_data"]["save_location"]

    Args:
        None

    Returns:
        None
    """
    logger.debug("Running the fetch_data function")

    #Loading the config
    with open(os.path.join("config","config.yml"), "r") as f:
        config = yaml.safe_load(f)

    #Starting the scraping process
    tstart = datetime.datetime.now()
    
    logger.info("Starting web scraping now.")
    for i in range(config["fetch_data"]["indices"]["start"], config["fetch_data"]["indices"]["end"]+1):
        try:
            req_link = config["fetch_data"]["link"] + str(i) + "/pg" + str(i) + ".txt"
            response = requests.get(req_link)
            
            response.encoding = "UTF-8"
            
            if response.status_code == 200:
                with open(config["fetch_data"]["save_location"] + str(i) + ".txt", "w", encoding="UTF-8") as text_file:
                    text_file.write(response.text)
            else:
                logger.error("Status Code {} returned for index {}".format(response.status_code, i))
            
            if i % 100 == 0:
                logger.info("At Index {}. Time Elapsed: {}".format(i, datetime.datetime.now()-tstart))  

        except Exception as e:
            logger.error(e)

    logger.info("All articles downloaded. Total Time taken: {}".format(datetime.datetime.now()-tstart))
    
    return