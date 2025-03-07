import importlib
import os
import logging
import json
import pickle
import pkgutil
import sys

import numpy as np

from abc import ABCMeta
from joblib import parallel_backend

from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans

from src.gpu_availability import is_cuda_available

cluster_dict = {}

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if is_cuda_available():
    from cuml.cluster import KMeans as CUMLKMeans
    

class ClusterMeta(ABCMeta):
    """
    Metaclass for tracking all subclasses of ClusteringModel.
    Automatically registers each subclass in the cluster_dict.
    """
    def __new__(cls, name, bases, attr):
        new_cls = super().__new__(cls, name, bases, attr)
        if name != 'ClusteringModel':
            cluster_dict[name.lower()] = new_cls
        return new_cls
    
    @classmethod
    def load_subclasses(cls, package_name):
        """Dynamically imports all modules in the package to register subclasses."""
        package = sys.modules[package_name]
        for _, modname, _ in pkgutil.iter_modules(package.__path__):
            importlib.import_module(f"{package_name}.{modname}")

class ClusteringModel(metaclass=ClusterMeta):
    """Wrapper to all clustering models"""
    def __init__(self, config, model):
        self.config = config
        self.model = model
        
    def save(self, clustering_folder):
        """Save clustering model to disk.

        Args:
            clustering_folder (str): Folder to save to.
        """
        
        os.makedirs(clustering_folder, exist_ok=True)
        with open(os.path.join(clustering_folder, "cluster_config.json"), "w", encoding="utf-8") as fout:
            fout.write(json.dumps(self.config))
        self.model.save(clustering_folder)
    
    @classmethod
    def load(cls, clustering_folder):
        """Load a saved clustering model from disk.

        Args:
            clustering_folder (str): Folder where `ClusteringModel` was saved to using `ClusteringModel.save`.

        Returns:
            ClusteringModel: The loaded object.
        """
        
        config_path = os.path.join(clustering_folder, "cluster_config.json")
        
        if not os.path.exists(config_path):
            config = {"type": "sklearnkmeans", 'kwargs': {}}
        else:
            with open(config_path, "r", encoding="utf-8") as fin:
                config = json.loads(fin.read())
                
        cluster_type = config.get("type", None)
        assert cluster_type is not None, f"{clustering_folder} is not a valid clustering folder"
        assert cluster_type in cluster_dict, f"invalid cluster type {config['type']}"
        model = cluster_dict[cluster_type].load(clustering_folder)
        return cls(config, model)
    
    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list or str): Training corpus in the form of a list of strings or path to text file.
            config (dict, optional): Dict with key `"type"` and value being the lower-cased name of the specific cluster
            class to use.
                Also contains keyword arguments to pass to the specified cluster. Default behavior is to use kmeans cluster with default arguments.
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            ClusteringModel: Trained cluster model.
        """
        
        config = config if config is not None else {"type": "sklearnkmeans", "kwargs": {}}
        LOGGER.debug(f"Train Clustering with config: {json.dumps(config, indent=True)}")
        cluster_type = config.get("type", None)
        assert(
            cluster_type is not None
        ), f"config {config} should contain a key 'type' for the cluster type" 
        model = cluster_dict[cluster_type].train(
            trn_corpus, config=config["kwargs"], dtype=dtype
        )
        config['kwargs'] = model.config
        return cls(config, model)
    
    @staticmethod
    def load_config_from_args(args):
        """Parse config from a `argparse.Namespace` object.

        Args:
            args (argparse.Namespace): Contains either a `cluster_config_path` (path to a json file) or `cluster_config_json` (a json object in string form).

        Returns:
            dict: The dict resulting from loading the json file or json object.

        Raises:
            Exception: If json object cannot be loaded.
        """
        
        if args.cluster_config_path is not None:
            with open(args.cluster_config_path, "r", encoding="utf-8") as fin:
                cluster_config_json = fin.read()
        else:
            cluster_config_json = args.cluster_config_json
        
        try:
            cluster_config = json.loads(cluster_config_json)
        except json.JSONDecodeError as jex:
            raise Exception(
                f"Failed to load clustering config json from {cluster_config_json} ({jex})"
            )
        return cluster_config
    
    @staticmethod
    def force_multi_core_processing_clustering_models(model, X_train):
        with parallel_backend('threading', n_jobs=-1):
            model.fit(X_train)
        return model 
    
class SklearnAgglomerativeClustering(ClusteringModel):
    """Hierarchical KMeans, Agglomerative Kmeans"""
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        
    def save(self, save_dir):
        """Save trained sklearn Agglomerative Clustering model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "clustering.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)
    
    @classmethod
    def load(cls, load_dir):
        """Load a saved sklearn Agglomerative Clustering model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            SklearnAgglmerativeClustering: The loaded object.
        """
        
        LOGGER.info(f"Loading Agglomerative Clustering model from {load_dir}")
        clustering_path = os.path.join(load_dir, "clustering.pkl")
        assert os.path.exists(clustering_path), f"clustering path {clustering_path} does not exist"
        
        with open(clustering_path, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model

    @classmethod
    def train(cls, trn_corpus, config={}, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to sklearn's AgglomerativeClustering.

        Returns:
            AgglomerativeClustering: Trained clustering.

        Raises:
            Exception: If `config` contains keyword arguments that the SklearnAgglomerativeClustering does not accept.
        """
        
        defaults = {
            'n_clusters': 2,
            'metric': 'euclidean',
            'memory': None,
            'connectivity': None,
            'compute_full_tree': 'auto',
            'linkage': 'ward',
            'distance_threshold': None,
            'compute_distances': False
            }
        
        try:
            config = {**defaults, **config}
            model = AgglomerativeClustering(**config)
        except TypeError:
            raise Exception(
                f"clustering config {config} contains unexpected keyword arguments for SklearnAgglomerativeClustering"
            )
        model = cls.force_multi_core_processing_clustering_models(model, trn_corpus)
        return cls(config, model)
    
    def get_params(self):
        return self.model.get_params()
    
class SklearnKMeans(ClusteringModel):
    """Simple KMeans"""
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        
    def save(self, save_dir):
        """Save trained sklearn KMeans model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "clustering.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)
    
    @classmethod
    def load(cls, load_dir):
        """Load a saved sklearn KMeans model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            SklearnKMeans: The loaded object.
        """
        
        LOGGER.info(f"Loading SklearnKmeans Clustering model from {load_dir}")
        clustering_path = os.path.join(load_dir, "clustering.pkl")
        assert os.path.exists(clustering_path), f"clustering path {clustering_path} does not exist"
        
        with open(clustering_path, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model

    @classmethod
    def train(cls, trn_corpus, config={}, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to sklearn's KMeans Clustering.

        Returns:
            KMeans: Trained clustering.

        Raises:
            Exception: If `config` contains keyword arguments that the SklearnKMeans does not accept.
        """
        
        defaults = {
            'n_clusters': 8,
            'init': 'k-means++',
            'n_init': 'auto',
            'max_iter': 300,
            'tol': 0.0001,
            'verbose': 0,
            'random_state': None,
            'copy_x': True,
            'algorithm': 'lloyd'
            }
        
        try:
            config = {**defaults, **config}
            model = KMeans(**config)
        except TypeError:
            raise Exception(
                f"clustering config {config} contains unexpected keyword arguments for SklearnKMeans Clustering"
            )
        model = cls.force_multi_core_processing_clustering_models(model, trn_corpus)
        return cls(config, model)
    
    def predict(self, predict_input):
        """Predict an input.

        Args:
            corpus (str, list): List of strings to predict.

        Returns:
            numpy.ndarray: Matrix of features.
        """
        return self.model.predict(predict_input)
    
    def get_params(self):
        return self.model.get_params()


class SklearnMiniBatchKMeans(ClusteringModel):
    """MiniBatchKmeans, Batching KMeans"""
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        
    def save(self, save_dir):
        """Save trained sklearn MiniBatchKMeans model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "clustering.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)
    
    @classmethod
    def load(cls, load_dir):
        """Load a saved sklearn MiniBatchKMeans model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            KMeans: The loaded object.
        """
        
        LOGGER.info(f"Loading SklearnMiniBatchKMeans Clustering model from {load_dir}")
        clustering_path = os.path.join(load_dir, "clustering.pkl")
        assert os.path.exists(clustering_path), f"clustering path {clustering_path} does not exist"
        
        with open(clustering_path, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model

    @classmethod
    def train(cls, trn_corpus, config={}, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to sklearn's MiniBatchKmeans.

        Returns:
            SklearnMiniBatchKMeans: Trained clustering.

        Raises:
            Exception: If `config` contains keyword arguments that the SklearnMiniBatchKmeans does not accept.
        """
        
        defaults = {
            'n_clusters': 8,
            'init': 'k-means++',
            'max_iter': 300,
            'batch_size': 1024,
            'verbose': 0,
            'compute_labels': True,
            'random_state': None,  
            'tol': 0.0,       
            'max_no_improvement': 10,
            'init_size': None,   
            'n_init': 'auto',
            'reassigment_ratio': 0.01
            }
        
        try:
            config = {**defaults, **config}
            model = MiniBatchKMeans(**config)
        except TypeError:
            raise Exception(
                f"clustering config {config} contains unexpected keyword arguments for SklearnMiniBatchKMeans Clustering"
            )
        model = cls.force_multi_core_processing_clustering_models(model, trn_corpus)
        return cls(config, model)
    
    def predict(self, predict_input):
        """Predict an input.

        Args:
            corpus (str, list): List of strings to predict.

        Returns:
            numpy.ndarray: Matrix of features.
        """
        return self.model.predict(predict_input)
    
    def get_params(self):
        return self.model.get_params()
    
class CumlKMeans(ClusteringModel):
    """Cuml KMeans with gpu support"""
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        
    def save(self, save_dir):
        """Save trained Cuml KMeans model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "clustering.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)
    
    @classmethod
    def load(cls, load_dir):
        """Load a saved Cuml KMeans model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            CumlKMeans: The loaded object.
        """
        
        LOGGER.info(f"Loading CumlKMeans Clustering model from {load_dir}")
        clustering_path = os.path.join(load_dir, "clustering.pkl")
        assert os.path.exists(clustering_path), f"clustering path {clustering_path} does not exist"
        
        with open(clustering_path, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model

    @classmethod
    def train(cls, trn_corpus, config={}, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to cuml's Kmeans.

        Returns:
            CumlKMeans: Trained clustering.

        Raises:
            Exception: If `config` contains keyword arguments that the CumlKmeans does not accept.
        """
        
        defaults = {
            'handle': None,
            'n_clusters': 8,
            'max_iter': 300,
            'tol': 0.0001,       
            'verbose': False,
            'random_state': 1,  
            'init': 'scalable-k-means++',
            'n_init': 1,
            'oversampling_factor': 2.0,
            'max_samples_per_batch': 32768,
            'convert_dtype': True,
            'output_type': None
            }
        
        try:
            config = {**defaults, **config}
            model = CUMLKMeans(**config)
        except TypeError:
            raise Exception(
                f"clustering config {config} contains unexpected keyword arguments for CumlKMeans Clustering"
            )
        model.fit(trn_corpus)
        return cls(config, model)

    def predict(self, predict_input):
        """Predict an input.

        Args:
            corpus (str, list): List of strings to predict.

        Returns:
            numpy.ndarray: Matrix of features.
        """
        return self.model.predict(predict_input)