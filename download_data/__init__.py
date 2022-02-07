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
    files = [tuple(x.split("\n")[0].split(":")) for x in f.readlines()]

#%%
def download_data(token, folder):

    """
        Parameters
        ----------
        token: str, token for data downloading
        path: str, folder containing downloaded data
    """

    # Creating data folder
    try:
        os.mkdir("./data")
    except:
        pass
    
    for file in files:
        file_path = f"{folder}/{file[0]}"


        # Download the file
        print(f"Downloading {file[0]}")
        download_file = True
        if os.path.exists(file_path) and os.path.getsize(file_path) == int(file[1]):
            download_file = False

        if download_file:
            uri = f"{file_uri}/{file[0]}/{token}"
            r = requests.get(uri, stream=True)
            download_size = int(int(r.headers.get('content-length'))/1000)

            with open(file_path, "wb") as f:
                with Bar(f"Downloading {file[0]}", max=int(download_size)) as bar:
                    for chunck in r.iter_content(chunk_size=1024):
                        f.write(chunck)
                        bar.next()

            # Unzip the file
            print(f"Unziping {file[0]}")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(folder)

            # Deleting the zip file
            print(f"Cleaning temporary files {file[0]}")
            os.remove(file_path)

    print("All done")


# %%
