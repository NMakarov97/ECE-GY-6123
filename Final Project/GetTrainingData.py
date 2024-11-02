import os
import gdown

def download(link: str) -> str:
    '''
    Download the given link and return the final directory

    Parameters:
    link: str of the URL of the file to be downloaded

    Returns:
    path: final path of the downloaded file
    '''
