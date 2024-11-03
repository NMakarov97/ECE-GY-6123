import os
import gdown

def main(link:str='https://drive.google.com/uc?id=1S72XYbK045eOAumbUHO5ITK_LvaUw_z6') -> str:
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

if __name__ == '__main__':
    main()
