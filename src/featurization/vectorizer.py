import os
import torch

import onnxruntime as ort
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer as TfidfVec
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

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
    
    @classmethod
    def export_to_onnx(cls, directory):
        """Exports BioBERT to ONNX if not already exported."""
        if os.path.exists(directory):
            print("ONNX model already exists. Skipping Export.")
            return 
        else:
            print("ONNX model doesn't exist. Exporting.")
        
        
        model = AutoModelForSequenceClassification.from_pretrained(cls.model_name)
        tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        
        # Create dummy input
        inputs = tokenizer("dummy test", return_tensors="pt")
        
      # Export to ONNX
        torch.onnx.export(
            model, 
            (inputs["input_ids"], inputs["attention_mask"]), 
            directory,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}},
            opset_version=11
        )
        print("Export complete.")
    
    @classmethod
    def predict_cpu(cls, corpus, directory):
        """Runs inference using ONNX for faster CPU execution"""
        tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        
        # Ensure ONNX model is exported before running inference
        cls.export_to_onnx(directory)
        
        session = ort.InferenceSession(directory)
        
        # Tokenize input
        inputs = tokenizer(corpus, return_tensors="pt")
        
        # Convert tensors to NumPy arrays
        onnx_inputs = {k: v.numpy() for k, v in inputs.items()}

        # Run inference
        return session.run(None, onnx_inputs)[0]
    
    @classmethod
    def predict_gpu(cls, corpus):
        tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        model = AutoModel.from_pretrained(cls.model_name)
        
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
