from sklearn.base import BaseEstimator, TransformerMixin
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

class QuestionMarkFeature(BaseEstimator, TransformerMixin):
    """Binary feature indicating whether there is a question mark in each element in X"""

    def __init__(self) -> None:
        super().__init__()

    def fit_transform(self, X, y, **fit_params):
        return self.transform(X)

    def fit(self, X, y, **fit_params):
        self.fitted = True
        return self

    def transform(self, X):
        transformed = []
        
        for text in list(X):
            transformed.append(int('?' in text))
        
        return np.asarray(transformed).reshape(-1, 1)

    
class TextLengthFeature(BaseEstimator, TransformerMixin):
    """Numeric features related to text length"""

    def __init__(self) -> None:
        self.stop_words = stopwords.words('english')
        super().__init__()

    def fit_transform(self, X, y, **fit_params):
        return self.transform(X)

    def fit(self, X, y, **fit_params):
        return self

    def transform(self, X):
        
        feats = list(pd.Series(X).map(self._get_num_feats).values)

        return np.asarray(feats)

    def _get_num_feats(self, text):
        out = [0] * 5
        words = word_tokenize(text)
        sents = sent_tokenize(text)

        # Number of characters
        out[0] = sum([len(w) for w in words])
        # Number of words
        out[1] = len(words)
        # Number of sentences
        out[2] = len(sents)            
        # Number of stop words
        out[3] = len([w for w in words if w in self.stop_words])
        # Avg word length
        out[4] = np.mean([len(w) for w in words]) if len(words) > 0 else 0

        return out