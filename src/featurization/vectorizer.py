from abc import ABCMeta
import json
import logging
import os
import pickle
import torch
import glob

import onnxruntime as ort
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from scipy.sparse import csr_matrix

from src.featurization.preprocessor import Preprocessor


vectorizer_dict = {}

LOGGER = logging.getLogger(__name__)

class VectorizerMeta(ABCMeta):
    """Metaclass for keeping track of all 'Vectorizer' subclasses"""
    def __new__(cls, name, bases, attr):
        new_cls = super().__new__(cls, name, bases, attr)
        if name != 'Vectorizer':
            vectorizer_dict[name.lower()] = new_cls
        return new_cls

class Vectorizer(metaclass=VectorizerMeta):
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        
    def save(self, vectorizer_folder):
        os.makedirs(vectorizer_folder, exist_ok=True)
        with open(os.path.join(vectorizer_folder, "config.json"), "w", encoding="utf-8") as fout:
            fout.write(json.dumps(self.config))
        self.model.save(vectorizer_folder)
        
    @classmethod
    def load(cls, vectorizer_folder):
        config_path = os.path.join(vectorizer_folder, "config.json")
        
        if not os.path.exists(config_path):
            config = {"type": "tfidf", 'kwargs': {}}
        else:
            with open(config_path, "r", encoding="utf-8") as fin:
                config = json.loads(fin.read())
                
        vectorizer_type = config.get("type", None)
        assert vectorizer_type is not None, f"{vectorizer_folder} is not a valid vectorizer folder"
        assert vectorizer_type in vectorizer_dict, f"invalid vectorizer type {config['type']}"
        model = vectorizer_dict[vectorizer_type].load(vectorizer_folder)
        return cls(config, model)

    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
        config = config if config is not None else {"type": "tfidf", "kwargs": {}}
        LOGGER.debug(f"Train Vectorizer with config: {json.dumps(config, indent=True)}")
        vectorizer_type = config.get("type", None)
        assert(
            vectorizer_type is not None
        ), f"config {config} should contain a key 'type' for the vectorizer type" 
        assert(
            isinstance(trn_corpus, list) or vectorizer_type == "tfidf"
        ), "only tfidf support from file training"
        model = vectorizer_dict[vectorizer_type].train(
            trn_corpus, config=config["kwargs"], dtype=dtype
        )
        return cls(config, model)
    
    def predict(self, corpus, **kwargs):
        if isinstance(corpus, str) and self.config["type"] != "tfidf":
            raise ValueError("Iterable over raw text expected for vectorizer other than tfidf.")
        return self.model.predict(corpus, **kwargs)

    @staticmethod
    def load_config_from_args(args):
        if args.vectorizer_config_path is not None:
            with open(args.vectorizer_config_path, "r", encoding="utf-8") as fin:
                vectorizer_config_json = fin.read()
        else:
            vectorizer_config_json = args.vectorizer_config_json
        
        try:
            vectorizer_config = json.loads(vectorizer_config_json)
        except json.JSONDecodeError as jex:
            raise Exception(
                "Failed to load vectorizer config json from {} ({})".format(
                    vectorizer_config_json, jex
                )
            )
        return vectorizer_config
        

class Tfidf(Vectorizer):
    
    def __init__(self, model=None):
        self.model = model
        
    def __del__(self):
        """Destruct self model instance"""
        self.model = None
    
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "vectorizer.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)
    
    @classmethod
    def load(cls, load_dir):
        vectorizer_path = os.path.join(load_dir, "vectorizer.pkl")
        assert os.path.exists(vectorizer_path), "vectorizer path {} does not exist".format(
            vectorizer_path
        )
        with open(vectorizer_path, "rb") as fvec:
            return cls(pickle.load(fvec))

    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
        
        defaults = {
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
        
        try:
            model = TfidfVectorizer(**{**defaults, **config})
        except TypeError:
            raise Exception(
                f"vectorizer config {config} contains unexpected keyword arguments for TfidfVectorizer"
            )
        model.fit(trn_corpus)
        return cls(model)
    
    def predict(self, corpus):
        return self.model.transform(corpus)
