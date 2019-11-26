"""
Author: Saurabh Annadate

This Script contains all helper functions

"""

import os

def create_dir(path):
    """Checks if a directory exists and creates if doesnt
    """
    if os.path.exists(path):
        return
    else:
        os.mkdir(path)
        return

