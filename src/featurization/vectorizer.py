import os
import torch

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer as TfidfVec
from transformers import AutoTokenizer, AutoModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from src.featurization.preprocessor import Preprocessor


class TfidfVectorizer(Preprocessor):
    
    @classmethod
    def train(cls, trn_corpus, dtype=np.float32):
        # min_df = 0.0
        # max df = 0.98

        # 
        # Mine -> (13298, 13715) min_df = 0.0001, max_df = 0.98
        x_linker_params = {
            "ngram_range": (1, 2),       # n-grams from 1 to 2
            "max_features": None,        # No max feature limit
            "min_df": 0.0,            # Minimum document frequency ratio
            "max_df": 0.98,                 # Maximum document frequency ratio
            "binary": False,             # Term frequency is not binary
            "use_idf": True,             # Use inverse document frequency
            "smooth_idf": True,          # Apply smoothing to idf
            "sublinear_tf": False,       # Use raw term frequency
            "norm": "l2",                # Apply L2 normalization
            "analyzer": "word",          # Tokenizes by word
            "stop_words": None,          # No stop words used
            "dtype": dtype
        }
        default = {

        }
        try:
            model = TfidfVec(**default)
        except TypeError:
            raise Exception(
                f"vectorizer config {x_linker_params} contains unexpected keyword arguments for TfidfVectorizer"
            )
        model.fit(trn_corpus)
        return cls(model=model, model_type='TfidfSkLearn')
    
    def predict(self, corpus):
        return self.model.transform(corpus)
    
class BioBertVectorizer(Preprocessor):
    
    model_name = "dmis-lab/biobert-base-cased-v1.2"
    
    def predict(cls, corpus):
        model_name = "dmis-lab/biobert-base-cased-v1.2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        inputs = tokenizer(corpus, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.squeeze(0).numpy() 
        
"""
class DistilBertVectorizer():

    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    @classmethod
    def predict(cls, corpus):
        inputs = cls.tokenizer(corpus, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = cls.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1) 
"""
