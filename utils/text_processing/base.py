#%%
from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd

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

#%%
def get_documents_embeddings (y, embedder, column):
    """
        Given a Dataframe containing study_id and a text column, return a numpy array of embeddings
        The idea of this function is to prevent to embed two times the same text (for computation efficiency)
        
        Parameters:
        -----------
        y: Dataframe containing study_id, and a text column
        embedder: Object of embedding creator containing a transform function
        column: column containing the text to Embed

        Output:
        -------
        Numpy array of size (n, embedding_size)
    """

    # Getting reports DF
    reports_df = y[["study_id", column]].fillna("").drop_duplicates("study_id").reset_index(drop=True)
    reports_list = reports_df[column].astype(str).values

    # Getting BERT embeddings
    reports_embeddings = embedder.fit_transform(reports_list)

    output = pd.merge(
        y[["study_id"]],
        reports_df[["study_id"]].join(
            pd.DataFrame(reports_embeddings)
        ),
        left_on="study_id",
        right_on="study_id",
        how="left"
    ).iloc[:,1:].values

    return output