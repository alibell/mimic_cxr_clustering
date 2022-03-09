import fasttext
import os
from utils.downloader import download, extract_gzip

FASTEXT_MODEL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
FASTEXT_MODEL_FILENAME = "fastext.cc.en.300.bin"

def download_fastext(output_folder):
    """
        Download and extract the fastext model

        Parameters:
        -----------
        output_folder: str, path where to save the files

        Output:
        -------
        model_path
    """

    model_path = f"{output_folder}/{FASTEXT_MODEL_FILENAME}"

    if os.path.exists(model_path) == False:
        gzip_path = f"{output_folder}/fastext.cc.en.300.bin.gz"

        print("Downloading model")
        download(FASTEXT_MODEL, gzip_path)

        print("Extraction")
        extract_gzip(gzip_path, model_path, remove_original=True)
    
    return model_path