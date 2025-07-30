import os
import joblib
import pickle

import numpy as np

from datetime import datetime

from xmr4el.featurization.label_embedding_factory import LabelEmbeddingFactory
from xmr4el.featurization.preprocessor import Preprocessor
from xmr4el.featurization.text_encoder import TextEncoder
from xmr4el.xmr.base import HierarchicaMLModel


class XModel():
    
    def __init__(self, 
                 vectorizer_config=None,
                 transformer_config=None,
                 dimension_config=None,
                 clustering_config=None,
                 matcher_config=None,
                 reranker_config=None,
                 depth=1
                 ):
        
        self.vectorizer_config = vectorizer_config
        self.transformer_config = transformer_config
        self.dimension_config = dimension_config
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        self.reranker_config = reranker_config
        self.depth = depth
        
        self._text_encoder = None
        self._hml = None
        
    @property
    def text_encoder(self):
        return self._text_encoder
    
    @text_encoder.setter
    def text_encoder(self, value):
        self._text_encoder = value

    @property
    def hierarchical_model(self):
        return self._hml
    
    @hierarchical_model.setter
    def hierarchical_model(self, value):
        self._hml = value
        
    def save(self, save_dir):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(save_dir, f"{self.__class__.__name__.lower()}_{timestamp}")
        os.makedirs(save_dir, exist_ok=False)
    
        state = self.__dict__.copy()
        
        model = self.hierarchical_model
        model_path = os.path.join(save_dir, f"hml")
        
        if model is not None:
            if hasattr(model, "save"):
                model.save(model_path)
            else:
                try:
                    joblib.dump(model, f"{model_path}.joblib")
                except ImportError:
                    with open(f"{model_path}.pkl", "wb") as f:
                        pickle.dump(model, f)  
            
            state.pop("_hml", None) # Popped _hml from class
        
        text_encoder = self.text_encoder
        text_encoder_path = os.path.join(save_dir, f"text_encoder")
        
        if text_encoder is not None:
            if hasattr(text_encoder, "save"):
                text_encoder.save(text_encoder_path)
            else:
                try:
                    joblib.dump(text_encoder, f"{text_encoder_path}.joblib")
                except ImportError:
                    with open(f"{text_encoder_path}.pkl", "wb") as f:
                        pickle.dump(text_encoder, f)  
                          
        with open(os.path.join(save_dir, "xmodel.pkl"), "wb") as fout:
            pickle.dump(state, fout)
    
    @classmethod
    def load(cls, load_dir):
        xmodel_path = os.path.join(load_dir, "xmodel.pkl")
        assert os.path.exists(xmodel_path), f"XModel path {xmodel_path} does not exist"
        
        with open(xmodel_path, "rb") as fin:
            model_data = pickle.load(fin)
            
        model = cls()
        model.__dict__.update(model_data)
        
        model_path = os.path.join(load_dir, "hml")
        hml = HierarchicaMLModel.load(model_path)
        setattr(model, "_hml", hml)
        
        text_encoder_path = os.path.join(load_dir, "text_encoder")
        text_encoder = TextEncoder.load(text_encoder_path)
        setattr(model, "_text_encoder", text_encoder)
        
        return model
    
    def _fit(self, X_text, Y_text):
        """Returns embeddings: ndarray"""
        
        X_processed, Y_label_to_indices = Preprocessor.prepare_data(X_text, Y_text)
        
        # Encode X_processed
        text_encoder = TextEncoder(
            vectorizer_config=self.vectorizer_config,
            transformer_config=self.transformer_config,
            dimension_config=self.dimension_config, 
            flag=1 # Needs to be a variable, could have a stop to check
            )
        
        self.text_encoder = text_encoder
        
        X_emb = text_encoder.encode(X_processed)
        
        Y_label_matrix = LabelEmbeddingFactory.generate_label_matrix(Y_label_to_indices)
        
        # Process Labels
        Y_binazer, _ = LabelEmbeddingFactory.label_binarizer(Y_label_matrix)
        Z = LabelEmbeddingFactory.generate_PIFA(X_emb, Y_binazer)
        
        return X_emb, Y_binazer, Z 
    
    def train(self, X_text, Y_text):
        
        X, Y, Z = self._fit(X_text=X_text, Y_text=Y_text)
        
        label_indices = np.arange(Z.shape[0])
        global_to_local_indices = {global_idx: local_idx for local_idx, global_idx in enumerate(label_indices)}
        
        hml = HierarchicaMLModel(clustering_config=self.clustering_config, 
                                 matcher_config=self.matcher_config, 
                                 reranker_config=self.reranker_config, 
                                 layer=self.depth)
        
        hml.train(X_train=X, 
                  Y_train=Y, 
                  Z_train=Z, 
                  local_label_indices=label_indices,
                  global_to_local_indices=global_to_local_indices)
        
        self.hierarchical_model = hml
        del hml
        print(self.hierarchical_model)
        
    def predict(self, X_input):
        pass
        
        
        
    
    
        