"""
    Script for file downloader
"""

from urllib.request import urlopen
from tqdm import tqdm
import gzip
import os

def download(uri, output_file):
    """
        download : download a file with a nice progress bar

        Parameters:
        -----------
        uri: uri of the file to download
        output_file: path of the downloaded file
    """

    # Getting file size
    uri_open = urlopen(uri)
    uri_metadata = uri_open.info()
    uri_size = int(uri_metadata.get("Content-Length"))

    chunk_size = 1024**2
    n_chunk = ((uri_size//chunk_size)+((uri_size%chunk_size) != 0))

    with open(output_file, "wb") as file:
        for i in tqdm(range(n_chunk)):
            chunk = uri_open.read(chunk_size)
            file.write(chunk)

def extract_gzip(file, output_file, remove_original=False):
    """
        Simple function which extract the content of an archive
        Supporter archives :
        - gz

        Parameters:
        -----------
        file: file to extract
        output_file: destination of extracted file
        remove_original: if set to True the original files are removed
    """

    compressed_file = gzip.open(file, "rb")
    chunk_size = 1024**2

    extracted = False
    with open(output_file, "wb") as outfile_file_stream:
            while True:
                chunk = compressed_file.read(chunk_size)
                outfile_file_stream.write(chunk)

                if len(chunk) == 0:
                    extracted = True
                    break

    if remove_original and extracted:
        os.remove(file)