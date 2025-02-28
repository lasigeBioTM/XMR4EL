import os
import logging
import json
import glob
import pickle
import shutil
import subprocess
import torch

import onnxruntime as ort
import numpy as np

from abc import ABCMeta

from transformers import AutoTokenizer, AutoModel


vectorizer_dict = {}

LOGGER = logging.getLogger(__name__)

class BertVectorizerMeta(ABCMeta):
    """Metaclass for keeping track of all 'Vectorizer' subclasses"""
    def __new__(cls, name, bases, attr):
        new_cls = super().__new__(cls, name, bases, attr)
        if name != 'BertVectorizer':
            vectorizer_dict[name.lower()] = new_cls
        return new_cls

class BertVectorizer(metaclass=BertVectorizerMeta):
    
    def __init__(self, model_name, config, emb):
        self.model_name = model_name
        self.config = config
        self.emb = emb
        
    def save(self, vectorizer_folder):
        os.makedirs(vectorizer_folder, exist_ok=True)
        with open(os.path.join(vectorizer_folder, "config.json"), "w", encoding="utf-8") as fout:
            fout.write(json.dumps(self.config))
        self.model.save(vectorizer_folder)
        
    @classmethod
    def load(cls, vectorizer_folder):
        config_path = os.path.join(vectorizer_folder, "config.json")
        
        if not os.path.exists(config_path):
            config = {"type": "biobert", 'kwargs': {}}
        else:
            with open(config_path, "r", encoding="utf-8") as fin:
                config = json.loads(fin.read())
                
        vectorizer_type = config.get("type", None)
        assert vectorizer_type is not None, f"{vectorizer_folder} is not a valid vectorizer folder"
        assert vectorizer_type in vectorizer_dict, f"invalid vectorizer type {config['type']}"
        model = vectorizer_dict[vectorizer_type].load(vectorizer_folder)
        return cls(model_name=config.get("type", "biobert"), config=config, emb=model)


    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
        config = config if config is not None else {"type": "biobert", "kwargs": {}}
        LOGGER.debug(f"Train Vectorizer with config: {json.dumps(config, indent=True)}")
        vectorizer_type = config.get("type", None)
        assert(
            vectorizer_type is not None
        ), f"config {config} should contain a key 'type' for the vectorizer type" 
        assert(
            isinstance(trn_corpus, list)
        ), "No model supports from file training"
        model = vectorizer_dict[vectorizer_type].train(
            trn_corpus, config=config["kwargs"], dtype=dtype
        )
        return cls(config, model)

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

    @staticmethod
    def is_cuda_available():
        # Default to using CPU models and set gpu_available to False
        gpu_available = False

        # Check if nvidia-smi is available on the system
        if shutil.which('nvidia-smi'):
            try:
                # Run the nvidia-smi command to check for an NVIDIA GPU
                subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                gpu_available = True
            except subprocess.CalledProcessError as e:
                # Print the error message and continue execution
                print(f"GPU acceleration is unavailable: {e}. Defaulting to CPU models.")
        else:
            print("nvidia-smi command not found. Assuming no NVIDIA GPU.")

        return gpu_available

    
class BioBert(BertVectorizer):
    
    model_name = "dmis-lab/biobert-base-cased-v1.2"
    
    def __init__(self, labels=None):
        self.labels = labels
        
        self.model_name = BioBert.model_name
        
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "vectorizer.pkl"), "wb") as fout:
            pickle.dump(self.__dict__, fout)
    
    @classmethod
    def load(cls, load_dir):
        vectorizer_path = os.path.join(load_dir, "vectorizer.pkl")
        assert os.path.exists(vectorizer_path), "vectorizer path {} does not exist".format(
            vectorizer_path
        )
        with open(vectorizer_path, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model
    
    @classmethod
    def train(cls, trn_corpus, config, dtype=np.float32):
        
        defaults = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": 512,
            "batch_size": 400,
            "onnx_directory": None
        }
        
        try:
            gpu_availability = cls.is_cuda_available()
            
            config = {**defaults, **config}
            
            if gpu_availability:
                config.pop("onnx_directory", None)
                cls.predict_gpu(trn_corpus, dtype, **config)
            else:
                cls.predict_cpu(trn_corpus, dtype, **config)
        except TypeError:
            raise Exception(
                f"vectorizer config {config} contains unexpected keyword arguments for TfidfVectorizer"
            )
    
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
    def predict_cpu(cls, corpus, dtype=np.float32, return_tensors="pt", padding=True, truncation=True, max_length=512, batch_size=400, onnx_directory=None):
        if onnx_directory is None:
            raise ValueError("ONNX directory must be specified for CPU inference.")
        
        output_prefix = "onnx_embeddings"
        
        """Runs inference using ONNX for faster CPU execution"""
        tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        
        # Ensure ONNX model is exported before running inference
        cls.export_to_onnx(onnx_directory)
        
        session = ort.InferenceSession(onnx_directory)
        
        #Split corpus into smaller batches 
        num_batches = len(corpus) // batch_size + (1 if len(corpus) % batch_size != 0 else 0)
        print(num_batches)
        
        for batch_idx in range(num_batches):
            print(f"Number of the batch: {batch_idx}")
            
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(corpus))
            batch = corpus[start:end]
            
            # Tokenize input
            inputs = tokenizer(batch, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length)
            
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
        all_embeddings = np.concatenate([np.load(f)["embeddings"] for f in batch_files], axis=0).astype(dtype)
        
        # Remove all batch files after saving the final file
        for f in batch_files:
            os.remove(f)
        
        return cls(all_embeddings)
        
    @classmethod
    def predict_gpu(cls, corpus, dtype=np.float32, return_tensors="pt", padding=True, truncation=True, max_length=512, batch_size=400):
        output_prefix = "gpu_embeddings"
        
        tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        model = AutoModel.from_pretrained(cls.model_name)  
        model.eval()  # Set to inference mode
        model.to("cuda")
        
        num_batches = len(corpus) // batch_size + (1 if len(corpus) % batch_size != 0 else 0)
        print(f"Total batches: {num_batches}")

        for batch_idx in range(num_batches):
            print(f"Processing batch -> {batch_idx}")

            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(corpus))
            batch = corpus[start:end]

            # Tokenize & move input tensors to GPU
            inputs = tokenizer(batch, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length).to("cuda")

            with torch.no_grad():
                outputs = model(**inputs)  # Forward pass on GPU
                batch_results = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Move results to CPU

            batch_filename = f"{output_prefix}_batch{batch_idx}.npz"
            np.savez_compressed(batch_filename, embeddings=batch_results)

        # Load and concatenate all batch embeddings
        batch_files = sorted(glob.glob(f"{output_prefix}_batch*.npz"))
        all_embeddings = np.concatenate([np.load(f)["embeddings"] for f in batch_files], axis=0).astype(dtype)

        # Clean up batch files
        for f in batch_files:
            os.remove(f)

        return cls(all_embeddings)