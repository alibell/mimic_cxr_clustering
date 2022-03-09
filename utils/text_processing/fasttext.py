#%%
import fasttext
import tempfile
import os
from sklearn.utils import check_array
import numpy as np

from ..downloader import download, extract_gzip
from .base import Embedder
#%%
FASTEXT_MODEL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
FASTEXT_MODEL_FILENAME = "fastext.cc.en.300.bin"

#%%
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
# %%
class FastextEmbedder(Embedder):
    def __init__ (self, preprocessing=True, model_folder=None):
        """
            Parameters:
            -----------
            preprocessing; boolean, wether to preprocess or not the text
            model_folder: str, path where to store the embedding model, if None a temporary folder is used
        """
        super().__init__(preprocessing=preprocessing)

        # Getting fasttext
        if model_folder is None:
            self.model_folder = tempfile.TemporaryDirectory().name
        else:
            self.model_folder = model_folder
        self.fasttext_model_path = download_fastext(self.model_folder)

        # Loading fasttext
        self.fasttext_model = fasttext.load_model(self.fasttext_model_path)

    def fit (self, X, y=None):
        """
            Parameters:
            X: Numpy array of size (n_samples, ) containing the text to preprocess
            y: No y expected here, added for sklearn compatibility
        """
        
        return self

    def transform (self, X, y=None):
        """
            Parameters:
            X: Numpy array of size (n_samples, ) containing the text to preprocess
            y: No y expected here, added for sklearn compatibility
        """

        X = check_array(X, ensure_2d=False, dtype="str")
        X_embedded = np.array([self.fasttext_model.get_sentence_vector(x) for x in X])

        return X_embedded