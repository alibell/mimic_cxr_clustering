"""
    This script download a sample of radiography data from MIMIC CRX and generate a compressed lower resolution of them
"""

#%%
from random import random
import pycurl
import pandas as pd
import os
import glob

#%%
# Parameters
p_sample = 0.02
random_seed = 42

#%%
# MIMIC Parameters
mimic_uri = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/"
# %%
# Getting credentials
with open("./credentials") as f:
    username, password = [x.split("\n")[0] for x in f.readlines()]

# %%
# Function to download file
def download_file(username, password, uri, path, verbose=True):
    # Launch a curl session
    c = pycurl.Curl()
    c.setopt(pycurl.USERAGENT, 'Wget/1.13.4 (linux-gnu)')
    c.setopt(pycurl.USERPWD, f'{username}:{password}')
    if verbose:
        c.setopt(pycurl.NOPROGRESS, False)
    c.setopt(pycurl.URL, uri)

    with open(path, "wb") as f:
        print(f"Writting {path}")
        c.setopt(pycurl.WRITEDATA, f)
        c.perform()
        c.close()
# %%
# Getting the data file list
files = [
    "mimic-cxr-2.0.0-chexpert.csv.gz",
    "mimic-cxr-2.0.0-metadata.csv.gz",
    "mimic-cxr-2.0.0-negbio.csv.gz"
]

for file in files:
    file_uri = f"{mimic_uri}{file}"
    file_path = f"../data/{file}"
    download_file(username, password, file_uri, file_path)

# %%
# Getting sample of current list
chexpert_df = pd.read_csv(f"../data/{files[0]}")
chexpert_df_sample = chexpert_df \
                   .sample(frac=p_sample, replace=False, random_state=random_seed) \
                   .reset_index(drop=True)
chexpert_df_sample.to_csv(f"../data/{files[0]}", compression='gzip')

study_list = chexpert_df_sample["study_id"].unique().tolist()

for file in files[1:]:
    file_df = pd.read_csv(f"../data/{file}")
    file_df_sample = file_df[
        file_df["study_id"].isin(study_list)
    ].reset_index(drop=True)

    file_df_sample.to_csv(f"../data/{file}", compression='gzip')


# %%
# Downloading files
metadata_df = pd.read_csv(f"../data/{files[1]}")
metadata_df["subject_id_str"] = metadata_df["subject_id"].astype("str")
metadata_df["study_id_str"] = metadata_df["study_id"].astype("str")

paths = ("files/p"+metadata_df["subject_id_str"].str.slice(0,2)+"/"+ \
    "p"+metadata_df["subject_id_str"]+"/"+ \
    "s"+metadata_df["study_id_str"]+"/" + \
    metadata_df["dicom_id"] + ".jpg").tolist()

for path in paths:
    subfolder = "/".join(path.split("/")[0:-1])
    folder = f"../data/{subfolder}"
    filename = path.split("/")[-1]
    filepath = f"{folder}/{filename}"
    fileuri = f"{mimic_uri}{path}"

    # Creating folder
    os.makedirs(folder, exist_ok=True)

    # Getting file
    print(f"Downloading {filename}")
    download_file(username, password, fileuri, filepath, verbose=False)

# %%
# Converting files to JPG