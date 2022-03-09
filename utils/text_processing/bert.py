#%%
from transformers import BertTokenizer, BertModel
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils import check_array

from .base import Embedder

#%%
BERT_MODEL = "bert-base-uncased"

#%%
class BertEmbedder(Embedder):
    def __init__ (self, preprocessing=True, batch_size=64, use_gpu_if_available=True):
        """
            Parameters:
            -----------
            preprocessing; boolean, wether to preprocess or not the text
            batch_size: int, size of the batch sent to bert
            use_gpu_if_available: boolean, if true the data will be sent to the GPU for inference
        """
        super().__init__(preprocessing=preprocessing)

        self.batch_size = batch_size
        self.use_gpu_if_available = use_gpu_if_available

        # Setting device
        if self.use_gpu_if_available:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        
        # Loading bert model
        self.bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self.bert_model = BertModel.from_pretrained(BERT_MODEL)

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
        n_samples = X.shape[0]
        n_batch = (n_samples//self.batch_size) + ((n_samples%self.batch_size) != 0)

        embeddings = []
        bert_model = self.bert_model.to(self.device)

        for i in range(n_batch):
            tokens = pad_sequence([
                torch.tensor(self.bert_tokenizer.encode(x), dtype=torch.int32) 
                for x in X[i*self.batch_size:(i+1)*self.batch_size]], batch_first=True, padding_value=self.bert_tokenizer.pad_token_id)
            tokens = tokens.to(self.device)
            with torch.no_grad():
                embedding = bert_model(tokens)[0]
                embedding = embedding.mean(axis=1)
                embeddings.append(
                    embedding
                )
        embeddings = torch.concat(embeddings, axis=0)
        embeddings = embeddings.cpu().numpy()

        return embeddings