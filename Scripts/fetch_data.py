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
import time

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
    with open(os.path.join("Config","config.yml"), "r") as f:
        config = yaml.safe_load(f)

    #Starting the scraping process
    tstart = datetime.datetime.now()
    err_count = 0

    logger.info("Starting web scraping now.")
    for i in range(config["fetch_data"]["indices"]["start"], config["fetch_data"]["indices"]["end"]+1):
        try:
            time.sleep(1)
            req_link1 = "http://www.gutenberg.org/cache/epub/" + str(i) + "/pg" + str(i) + ".txt"
            response1 = requests.get(req_link1)
            
            req_link2 = "http://www.gutenberg.org/files/" + str(i) + "/" + str(i) + "-0.txt"
            response2 = requests.get(req_link2)
            
            response1.encoding = "UTF-8"
            response2.encoding = "UTF-8"
            
            if response1.status_code == 200:
                with open(config["fetch_data"]["save_location"] + str(i) + ".txt", "w", encoding="UTF-8") as text_file:
                    text_file.write(response1.text)
                
            elif response2.status_code == 200:
                with open(config["fetch_data"]["save_location"] + str(i) + ".txt", "w", encoding="UTF-8") as text_file:
                    text_file.write(response2.text)
                
            else:
                err_count = err_count + 1   
                logger.error("Status Code {} returned for index {}".format(response.status_code, i))
            
            if i % 500 == 0:
                time.sleep(30)
                logger.info("At Index {}. Time Elapsed: {}".format(i, datetime.datetime.now()-tstart))  

        except Exception as e:
            logger.error(e)
    
    logger.info("Total Errorred documents: {}".format(err_count))
    logger.info("Total Successful documents: {}".format(config["fetch_data"]["indices"]["end"] - config["fetch_data"]["indices"]["start"] + 1 -err_count))
    logger.info("Total Time taken: {}".format(datetime.datetime.now()-tstart))

    return
