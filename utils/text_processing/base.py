#%%
from sklearn.base import BaseEstimator,TransformerMixin

#%%
class Embedder(BaseEstimator,TransformerMixin):
    """
        Embedder object
        Class of object which produce for a given text and embedding
        This class is in a sklearn format with a fit and transforma function
    """

    def __init__ (self, preprocessing=True):
        """
            Parameters:
            -----------
            preprocessing; boolear, wether to preprocess or not the text
        """

        self.preprocessing = preprocessing

    def _preprocessing (self, X):
        """
            Parameters:
            -----------
            X: Numpy array of size (n_samples, ) containing the text to pre-process
        """

        pass

    def fit (self, X, y=None):
        """
            Parameters:
            X: Numpy array of size (n_samples, ) containing the text to preprocess
            y: No y expected here, added for sklearn compatibility
        """

        raise NotImplementedError

    def transform (self, X, y=None):
        """
            Parameters:
            X: Numpy array of size (n_samples, ) containing the text to preprocess
            y: No y expected here, added for sklearn compatibility
        """

        raise NotImplementedError