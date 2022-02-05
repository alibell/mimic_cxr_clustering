"""
    This script download the data from mimic-cxr local serve
"""

#%%
import requests
import pathlib
from progress.bar import Bar
import os
import zipfile

#%%
current_folder = pathlib.Path(__file__).parent.absolute()
file_uri = "http://chessm2ds.alibellamine.me:5000/cxr/"
with open(f"{current_folder}/metadata", "r") as f:
    file_name, file_size = f.readline().split(":")

#%%
def download_data(token, folder):

    """
        Parameters
        ----------
        token: str, token for data downloading
        path: str, folder containing downloaded data
    """

    file_path = f"{folder}/{file_name}"

    # Download the file
    print("Downloading MIMIC-IV data")
    uri = f"{file_uri}/{token}"
    r = requests.get(uri, stream=True)
    file_size = int(int(r.headers.get('content-length'))/1000)

    with open(file_path, "wb") as f:
        with Bar(f"Downloading {file_name}", max=int(file_size)) as bar:
            for chunck in r.iter_content(chunk_size=1024):
                f.write(chunck)

    # Unzip the file
    print("Unziping downloaded file")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(folder)

    # Deleting the zip file
    print("Cleaning temporary files")
    os.remove(file_path)

    print("All done")
