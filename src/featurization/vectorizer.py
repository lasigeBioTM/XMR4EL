import os
import torch

import onnxruntime as ort
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer as TfidfVec
from transformers import AutoTokenizer, AutoModel

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
        
        
        model = AutoModel.from_pretrained(cls.model_name)
        tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        
        # Create dummy input
        dummy_test = ["This is a dummy test sentence for ONNX export."]
        inputs = tokenizer(dummy_test, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
      # Export to ONNX
        torch.onnx.export(
            model, 
            (inputs["input_ids"], inputs["attention_mask"]), 
            directory,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_length"}, 
                "attention_mask": {0: "batch", 1: "seq_length"}},
            opset_version=14
        )
        print("Export complete.")
    
    @classmethod
    def predict_cpu(cls, corpus, directory, batch_size=200, output_file="onnx_embeddings.npy"):
        """Runs inference using ONNX for faster CPU execution"""
        tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        
        # Ensure ONNX model is exported before running inference
        cls.export_to_onnx(directory)
        
        session = ort.InferenceSession(directory)
        
        # Create or clear the output file if it already exists
        if os.path.exists(output_file):
            print("File Already Exists, remove it or rename it")
            exit()
        
        #Split corpus into smaller batches 
        num_batches = len(corpus) // batch_size + (1 if len(corpus) % batch_size != 0 else 0)
        print(num_batches)
        all_results = []
        
        for i in range(num_batches):
            # Get the current batch
            batch = corpus[i * batch_size: (i + 1) * batch_size]
            
            # Tokenize input
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Convert tensors to NumPy arrays
            onnx_inputs = {
                "input_ids": inputs["input_ids"].numpy().astype(np.int64),
                "attention_mask": inputs["attention_mask"].numpy().astype(np.int64)
            }
            
            # Run inference for the current batch
            batch_results = session.run(None, onnx_inputs)[0]
            
            with open(output_file, "ab") as f:
                np.save(f, batch_results)
                
        embeddings_list = []
        with open(output_file, "rb") as f:
            while True:
                try:
                    embeddings_list.append(np.load(f))
                except ValueError:  # End of file reached
                    break
                
        # Convert list to NumPy array
        final_embeddings = np.concatenate(embeddings_list, axis=0)
        
        return final_embeddings
    
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
