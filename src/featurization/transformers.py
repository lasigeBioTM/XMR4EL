import os
import logging
import json
import glob
import pickle
import shutil
import torch

import onnxruntime as ort
import numpy as np

from pathlib import Path
from abc import ABCMeta
from transformers import AutoTokenizer, AutoModel

from src.gpu_availability import is_cuda_available


transformer_dict = {}

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TransformersMeta(ABCMeta):
    """Metaclass for keeping track of all 'Transformer' subclasses"""
    
    def __new__(cls, name, bases, attr):
        new_cls = super().__new__(cls, name, bases, attr)
        if name != 'Transformer':
            transformer_dict[name.lower()] = new_cls
        return new_cls

class Transformer(metaclass=TransformersMeta):
    """Wrapper class for all BERT-based Transformers."""
    
    def __init__(self, config, model):
        """Initialization

        Args:
            config (dict): Dict with key `"type"` and value being the lower-cased name of the specific transformer class to use.
                Also contains keyword arguments to pass to the specified transformer.
            model (Berttransformer): Trained Berttransformer.
        """
        
        self.config = config
        self.model = model
        
    def save(self, transformer_folder):
        """Save trained transformer to disk.

        Args:
            transformer_folder (str): Folder to save to.
        """
        
        LOGGER.info(f"Saving transformer to {transformer_folder}")
        os.makedirs(transformer_folder, exist_ok=True)
        with open(os.path.join(transformer_folder, "best_vec_config.json"), "w", encoding="utf-8") as fout:
            fout.write(json.dumps(self.config))
        self.model.save(transformer_folder)
        
    @classmethod
    def load(cls, transformer_folder):
        """Load a saved transformer from disk.

        Args:
            transformer_folder (str): Folder where `Berttransformer` was saved to using `Berttransformer.save`.

        Returns:
            Berttransformer: The loaded object.
        """
        
        LOGGER.info(f"Loading transformer from {transformer_folder}")
        config_path = os.path.join(transformer_folder, "bert_vec_config.json")
        
        if not os.path.exists(config_path):
            LOGGER.warning(f"Config file not found in {transformer_folder}, using default config")
            config = {"type": "biobert", 'kwargs': {}}
        else:
            with open(config_path, "r", encoding="utf-8") as fin:
                config = json.loads(fin.read())
                
        transformer_type = config.get("type", None)
        assert transformer_type is not None, f"{transformer_folder} is not a valid transformer folder"
        assert transformer_type in transformer_dict, f"invalid transformer type {config['type']}"
        model = transformer_dict[transformer_type].load(transformer_folder)
        return cls(config, model)


    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list or str): Training corpus in the form of a list of strings or path to text file.
            config (dict, optional): Dict with key `"type"` and value being the lower-cased name of the specific transformer class to use.
                Also contains keyword arguments to pass to the specified transformer. Default behavior is to use tfidf transformer with default arguments.
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            Berttransformer: Trained Berttransformer.
        """
        
        LOGGER.info("Starting training for Berttransformer")
        config = config if config is not None else {"type": "biobert", "kwargs": {}}
        
        transformer_type = config.get("type", None)
        assert transformer_type is not None, f"config {config} should contain a key 'type' for the transformer type" 
        # assert isinstance(trn_corpus, list), "No model supports from file training"
        
        LOGGER.info(f"Training transformer of type {transformer_type}")
        model = transformer_dict[transformer_type].train(
            trn_corpus, config=config["kwargs"], dtype=dtype
        )
        config["kwargs"] = model.config
        return cls(config, model)

    @staticmethod
    def load_config_from_args(args):
        """Parse config from a `argparse.Namespace` object.

        Args:
            args (argparse.Namespace): Contains either a `transformer_config_path` (path to a json file) or `transformer_config_json` (a json object in string form).

        Returns:
            dict: The dict resulting from loading the json file or json object.

        Raises:
            Exception: If json object cannot be loaded.
        """
        
        if args.transformer_config_path is not None:
            with open(args.transformer_config_path, "r", encoding="utf-8") as fin:
                transformer_config_json = fin.read()
        else:
            transformer_config_json = args.transformer_config_json
        
        try:
            transformer_config = json.loads(transformer_config_json)
        except json.JSONDecodeError as jex:
            raise Exception(
                "Failed to load transformer config json from {} ({})".format(
                    transformer_config_json, jex
                )
            )
        return transformer_config

    
class BioBert(Transformer):
    """BioBERT-based transformer."""
    
    model_name = "dmis-lab/biobert-base-cased-v1.2"
    
    def __init__(self, config=None, embeddings=None):
        """Initialization

        Args:
            config (dict): Dict with key `"type"` and value being the lower-cased name of the specific transformer class to use.
                Also contains keyword arguments to pass to the specified transformer.
            embeddings (numpy.ndarray): The Embeddings
            model_name (str): Transformer name
        """
        
        self.config = config
        self.embeddings = embeddings
        self.model_name = BioBert.model_name
        
    def save(self, save_dir):
        """Save trained tfidf transformer to disk.

        Args:
            save_dir (str): Folder to save the model.
        """
        
        LOGGER.info(f"Saving BioBERT transformer to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "transformer.pkl"), "wb") as fout:
            pickle.dump(self.__dict__, fout)
    
    @classmethod
    def load(cls, load_dir):
        """Load a BioBert Transformer from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            BioBert: The loaded object.
        """
        
        LOGGER.info(f"Loading BioBERT transformer from {load_dir}")
        transformer_path = os.path.join(load_dir, "transformer.pkl")
        assert os.path.exists(transformer_path), f"transformer path {transformer_path} does not exist"
        
        with open(transformer_path, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model
    
    @classmethod
    def train(cls, trn_corpus, config, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list of str or str): Training corpus in the form of a list of strings or path to corpus file/folder.
            config (dict): Dict with keyword arguments.
                The keywords are:
                    return_tensors (str, optional, default='pt'): If set, will return tensors instead of list of python integers.
                    padding (bool, str, optional, default=False): Activates and controls padding.
                    truncation (bool, str, optional, default=True): Activates and controls truncation. 
                    max_length (int, optional, default=512): Controls the maximum length to use by one of the truncation/padding parameters.
                    batch_size (int, optional): Number of entities for batch
                    output_prefix (str, optional): Name of the batch files
                    batch_dir (str, optional): Where the batch files will be saved  
                    onnx_directory (str): Location of the onnx model
            dtype (numpy.dtype): The data type to use. Default to `np.float32`.
            
        Notes: https://huggingface.co/docs/transformers/v4.20.0/main_classes/tokenizer
        Returns:
            config: Dict with all the parameters
            embeddings: The embedded labels
        """
        
        LOGGER.info("Training BioBERT transformer")
        defaults = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": 512,
            "batch_size": 400,
            "output_prefix": "onnx_embeddings",
            "batch_dir": "batch_dir",
            "onnx_directory": None
        }
        config = {**defaults, **config}
        gpu_availability = is_cuda_available()
        
        if gpu_availability:
            config.pop("onnx_directory", None)
            embeddings = cls.__predict_gpu(trn_corpus=trn_corpus, dtype=dtype, **config)
        else:
            embeddings = cls.__predict_cpu(trn_corpus=trn_corpus, dtype=dtype, **config)
        
        return cls(config, embeddings=embeddings)
    
    @classmethod
    def __export_to_onnx(cls, directory):
        """Exports BioBERT to ONNX if not already exported."""
        if os.path.exists(directory):
            LOGGER.info("ONNX model already exists. Skipping Export.")
            return 
        else:
            LOGGER.info("ONNX model doesn't exist. Exporting.")
        
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
        LOGGER.info("Export complete.")
    
    @classmethod
    def __predict_cpu(cls, trn_corpus, dtype=np.float32, return_tensors="pt", padding=True, truncation=True, max_length=512, batch_size=400, output_prefix="onnx_embeddings", batch_dir="batch_dir", onnx_directory=None):
        if onnx_directory is None:
            raise ValueError("ONNX directory must be specified for CPU inference.")
        
        batch_dir = f"{cls.__get_root_directory()}/{batch_dir}"
        emb_file = f"{batch_dir}/{output_prefix}" 
        
        """Runs inference using ONNX for faster CPU execution"""
        tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        
        # Ensure ONNX model is exported before running inference
        cls.__export_to_onnx(onnx_directory)
        
        session = ort.InferenceSession(onnx_directory)
        
        len_corpus = len(trn_corpus)
        
        cls.__create_batch_dir(batch_dir)
        
        #Split corpus into smaller batches 
        num_batches = len_corpus // batch_size + (1 if len_corpus % batch_size != 0 else 0)        
        LOGGER.info(f"Total of batches: {num_batches}.")
        
        for batch_idx in range(num_batches):
            LOGGER.info(f"Number of the batch: {batch_idx + 1}")
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len_corpus)
            batch = trn_corpus[start:end]
            
            print(batch)
            
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
            
            batch_filename = f"{emb_file}_batch{batch_idx}.npz"
            np.savez_compressed(batch_filename, embeddings=batch_results)  
                
        # After processing all batches, load the embeddings        
        batch_files = sorted(glob.glob(f"{emb_file}_batch*.npz"))
        
        # Concatenate all batches 
        all_embeddings = np.concatenate([np.load(f)["embeddings"] for f in batch_files], axis=0).astype(dtype)
        
        cls.__del_batch_dir(batch_dir)
        return all_embeddings
        
    @classmethod
    def __predict_gpu(cls, trn_corpus, dtype=np.float32, return_tensors="pt", padding=True, truncation=True, max_length=512, batch_size=400, output_prefix="onnx_embeddings", batch_dir="batch_dir"):
        batch_dir = f"{cls.__get_root_directory()}/{batch_dir}"
        emb_file = f"{batch_dir}/{output_prefix}" 
        
        tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        model = AutoModel.from_pretrained(cls.model_name)  
        model.eval()  # Set to inference mode
        model.to("cuda")
        
        len_corpus = len(trn_corpus)
        
        cls.__create_batch_dir(batch_dir)
        
        num_batches = len_corpus // batch_size + (1 if len_corpus % batch_size != 0 else 0)
        LOGGER.info(f"Total of batches: {num_batches}.")

        for batch_idx in range(num_batches):
            LOGGER.info(f"Number of the batch: {batch_idx + 1}")

            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len_corpus)
            batch = trn_corpus[start:end]

            # Tokenize & move input tensors to GPU
            inputs = tokenizer(batch, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length).to("cuda")

            with torch.no_grad():
                outputs = model(**inputs)  # Forward pass on GPU
                batch_results = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Move results to CPU

            batch_filename = f"{emb_file}_batch{batch_idx}.npz"
            np.savez_compressed(batch_filename, embeddings=batch_results)

        # Load and concatenate all batch embeddings
        batch_files = sorted(glob.glob(f"{emb_file}_batch*.npz"))
        all_embeddings = np.concatenate([np.load(f)["embeddings"] for f in batch_files], axis=0).astype(dtype)

        cls.__del_batch_dir(batch_dir)
        return all_embeddings
    
    def __create_batch_dir(batch_dir):
        """
        Create the batch dir if it does not exist,
        if it exists, remove any file inside 
        """

        if os.path.exists(batch_dir):
            for item in os.listdir(batch_dir):
                emb_path = os.path.join(batch_dir, item)
                os.remove(emb_path)
        else:
            LOGGER.warning(f"Directory does not exist, Creating")
            os.makedirs(batch_dir)
    
    def __del_batch_dir(batch_dir):
        """Delete the batch directory"""
        
        shutil.rmtree(batch_dir)
    
    @staticmethod
    def __get_root_directory():
        """Locate the src path of the project"""
        
        root_dir = Path(__file__).resolve().parent
        while not (root_dir / "src").exists() and root_dir != root_dir.parent:
            root_dir = root_dir.parent
        return root_dir