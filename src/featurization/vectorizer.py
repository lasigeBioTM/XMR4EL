import os
import torch
import glob

import onnxruntime as ort
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer as TfidfVec
from transformers import AutoTokenizer, AutoModel
from scipy.sparse import csr_matrix

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
    def predict_cpu(cls, corpus, directory, batch_size=400, output_prefix="onnx_embeddings"):
        """Runs inference using ONNX for faster CPU execution"""
        tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        
        # Ensure ONNX model is exported before running inference
        cls.export_to_onnx(directory)
        
        session = ort.InferenceSession(directory)
        
        #Split corpus into smaller batches 
        num_batches = len(corpus) // batch_size + (1 if len(corpus) % batch_size != 0 else 0)
        print(num_batches)
        
        for batch_idx in range(num_batches):
            print(f"Number of the batch: {batch_idx}")
            
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(corpus))
            batch = corpus[start:end]
            
            # Tokenize input
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Convert tensors to NumPy arrays
            onnx_inputs = {
                "input_ids": inputs["input_ids"].numpy().astype(np.int64),
                "attention_mask": inputs["attention_mask"].numpy().astype(np.int64)
            }
            
            # Run inference for the current batch
            batch_results = session.run(None, onnx_inputs)[0]
            
            batch_results = batch_results[:, 0, :]
            
            batch_filename = f"{output_prefix}_batch{batch_idx}.npz"
            np.savez_compressed(batch_filename, embeddings=batch_results)  
                
        # After processing all batches, load the embeddings        
        batch_files = sorted(glob.glob(f"{output_prefix}_batch*.npz"))
        
        # Concatenate all batches 
        all_embeddings = np.concatenate([np.load(f)["embeddings"] for f in batch_files], axis=0)
        
        # Convert the dense embeddings to a sparse matrix (CSR format)
        sparse_embeddings = csr_matrix(all_embeddings)
        
        # Remove all batch files after saving the final file
        for f in batch_files:
            os.remove(f)
            print(f"Deleted: {f}")
        
        return sparse_embeddings
        
    @classmethod
    def predict_gpu(cls, corpus):
        tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        model = AutoModel.from_pretrained(cls.model_name)
        
        inputs = tokenizer(corpus, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return csr_matrix(outputs.last_hidden_state.squeeze(0).numpy())
        

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
