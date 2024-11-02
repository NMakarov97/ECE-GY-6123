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

    path = "nyu_depth_v2.zip"
    if os.path.isfile(path):
        # Data zip file already exists
        return path
    else:
        # Download the file
        gdown.download(link, output=path)
        return path
