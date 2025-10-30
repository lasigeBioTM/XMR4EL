import importlib
import os
import json
import pickle
import pkgutil
import sys
import multiprocessing
import torch
import faiss

import numpy as np

from abc import ABCMeta
from joblib import parallel_backend
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from kmeans_pytorch import KMeans as PyTorchBalancedKMeans

# if torch.cuda.is_available():
#     from cuml.cluster import KMeans as CUMLKMeans


cluster_dict = {}


class ClusterMeta(ABCMeta):
    """
    Metaclass for tracking all subclasses of ClusteringModel.
    Automatically registers each subclass in the cluster_dict.
    """

    def __new__(cls, name, bases, attr):
        new_cls = super().__new__(cls, name, bases, attr)
        if name != "ClusteringModel":
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
        with open(
            os.path.join(clustering_folder, "cluster_config.json"),
            "w",
            encoding="utf-8",
        ) as fout:
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
            config = {"type": "sklearnkmeans", "kwargs": {}}
        else:
            with open(config_path, "r", encoding="utf-8") as fin:
                config = json.loads(fin.read())

        cluster_type = config.get("type", None)
        assert (
            cluster_type is not None
        ), f"{clustering_folder} is not a valid clustering folder"
        assert cluster_type in cluster_dict, f"invalid cluster type {config['type']}"
        model = cluster_dict[cluster_type].load(clustering_folder, config["kwargs"])
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

        config = (
            config if config is not None else {"type": "sklearnkmeans", "kwargs": {}}
        )
        # LOGGER.debug(f"Train Clustering with config: {json.dumps(config, indent=True)}")
        cluster_type = config.get("type", None)
        assert (
            cluster_type is not None
        ), f"config {config} should contain a key 'type' for the cluster type"
        model = cluster_dict[cluster_type].train(
            trn_corpus, config=config["kwargs"], dtype=dtype
        )
        config["kwargs"] = model.config
        return cls(config, model)

    def labels(self):
        return self.model.labels()

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
        with parallel_backend("threading", n_jobs=-1):
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
    def load(cls, load_dir, config):
        """Load a saved sklearn Agglomerative Clustering model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            SklearnAgglmerativeClustering: The loaded object.
        """

        # LOGGER.info(f"Loading Agglomerative Clustering Model from {load_dir}")
        clustering_path = os.path.join(load_dir, "clustering.pkl")
        assert os.path.exists(
            clustering_path
        ), f"clustering path {clustering_path} does not exist"

        with open(clustering_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls(config, model_data)
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
            "n_clusters": 2,
            "metric": "euclidean",
            "memory": None,
            "connectivity": None,
            "compute_full_tree": "auto",
            "linkage": "ward",
            "distance_threshold": None,
            "compute_distances": False,
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

    def labels(self):
        return self.model.labels_

    def get_params(self):
        return self.model.get_params()


class SklearnKMeans(ClusteringModel):
    """Simple KMeans"""

    def __init__(self, config=None, model=None):
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
    def load(cls, load_dir, config):
        """Load a saved sklearn KMeans model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            SklearnKMeans: The loaded object.
        """

        # LOGGER.info(f"Loading Sklearn Kmeans Clustering Model from {load_dir}")
        clustering_path = os.path.join(load_dir, "clustering.pkl")
        assert os.path.exists(
            clustering_path
        ), f"clustering path {clustering_path} does not exist"

        with open(clustering_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls(config, model_data)
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
            "n_clusters": 8,
            "init": "k-means++",
            "n_init": "auto",
            "max_iter": 300,
            "tol": 0.0001,
            "verbose": 0,
            "random_state": None,
            "copy_x": True,
            "algorithm": "lloyd",
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

    def labels(self):
        return self.model.labels_


class SklearnMiniBatchKMeans(ClusteringModel):
    """MiniBatchKmeans, Batching KMeans"""

    def __init__(self, config=None, model=None):
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
    def load(cls, load_dir, config):
        """Load a saved sklearn MiniBatchKMeans model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            KMeans: The loaded object.
        """

        # LOGGER.info(f"Loading Sklearn MiniBatchKMeans Clustering Model from {load_dir}")
        clustering_path = os.path.join(load_dir, "clustering.pkl")
        assert os.path.exists(
            clustering_path
        ), f"clustering path {clustering_path} does not exist"

        with open(clustering_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls(config, model_data)
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
            "n_clusters": 8,
            "init": "k-means++",
            "max_iter": 300,
            "batch_size": 0,  # Default: 1024
            "verbose": 0,
            "compute_labels": True,
            "random_state": None,
            "tol": 0.0,
            "max_no_improvement": 10,
            "init_size": None,
            "n_init": "auto",
            "reassignment_ratio": 0.01,
        }

        try:
            config = {**defaults, **config}

            if config["batch_size"] <= 0:
                num_cores = multiprocessing.cpu_count()
                batch_size = len(trn_corpus) // num_cores
                batch_size = max(1, batch_size)
                config["batch_size"] = batch_size

            model = MiniBatchKMeans(**config)
        except TypeError:
            raise Exception(
                f"clustering config {config} contains unexpected keyword arguments for SklearnMiniBatchKMeans Clustering"
            )
        model = model.fit(trn_corpus)
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

    def labels(self):
        return self.model.labels_


class CumlKMeans(ClusteringModel):
    """Cuml KMeans with gpu support"""

    def __init__(self, config=None, model=None):
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
    def load(cls, load_dir, config):
        """Load a saved Cuml KMeans model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            CumlKMeans: The loaded object.
        """

        # LOGGER.info(f"Loading Cuml KMeans Clustering Model from {load_dir}")
        clustering_path = os.path.join(load_dir, "clustering.pkl")
        assert os.path.exists(
            clustering_path
        ), f"clustering path {clustering_path} does not exist"

        with open(clustering_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls(config, model_data)
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
            "handle": None,
            "n_clusters": 8,
            "max_iter": 300,
            "tol": 0.0001,
            "verbose": False,
            "random_state": 1,
            "init": "scalable-k-means++",
            "n_init": 1,
            "oversampling_factor": 2.0,
            "max_samples_per_batch": 32768,
            "convert_dtype": True,
            "output_type": None,
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

    def labels(self):
        return self.model.labels_
    
    
class FaissKMeans(ClusteringModel):
    """Faiss Kmeans"""
    
    def __init__(self, config=None, model=None):
        self.model = model
        self.config = config
        self.centroids = None
        self.labels_ = None

    def save(self, save_dir):
        """Save trained FAISS KMeans model to disk.
        
        Args:
            save_dir (str): Folder to store serialized model.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS-compatible data (centroids + config)
        save_data = {
            'centroids': self.centroids,
            'labels': self.labels_,
            'faiss_index': faiss.serialize_index(self.model.index) if hasattr(self.model, 'index') else None
        }
        
        # Save model data
        with open(os.path.join(save_dir, "clustering.pkl"), "wb") as fout:
            pickle.dump(save_data, fout)

    @classmethod
    def load(cls, load_dir, config):
        """Load a saved Cuml KMeans model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            CumlKMeans: The loaded object.
        """

        # LOGGER.info(f"Loading FAISS KMeans Clustering Model from {load_dir}")
        clustering_path = os.path.join(load_dir, "clustering.pkl")
        assert os.path.exists(
            clustering_path
        ), f"clustering path {clustering_path} does not exist"

        with open(clustering_path, "rb") as fin:
            model_data = pickle.load(fin)

            # Reconstruct FAISS objects
        d = model_data['centroids'].shape[1]
        k = model_data['centroids'].shape[0]
        
        kmeans = faiss.Kmeans(
            d, k, 
            gpu=config.get('gpu', True),
            niter=config.get('max_iter', 300),
            nredo=config.get('n_init', 1)
        )
        kmeans.centroids = model_data['centroids']
        
        if model_data['faiss_index'] is not None:
            kmeans.index = faiss.deserialize_index(model_data['faiss_index'])
        
        # Create instance
        instance = cls(config, kmeans)
        instance.centroids = model_data['centroids']
        instance.labels_ = model_data['labels']
        
        return instance

    @classmethod
    def train(cls, trn_corpus, config={}, dtype=np.float32):
        """Train FAISS KMeans on a corpus.
        
        Args:
            trn_corpus: Training data (list of strings or numpy array).
            config: Configuration dictionary.
            dtype: Data type for embeddings.
            
        Returns:
            ClusteringModel: Wrapped trained model.
        """
        defaults = {
            "n_clusters": 8,
            "max_iter": 300,
            "tol": 1e-4,
            "verbose": False,
            "random_state": 42,
            "gpu": True,
            "nredo": 3,
            "spherical": False,
        }

        try:
            config = {**defaults, **config}
            # Train FAISS KMeans
            
            model = faiss.Kmeans(
                d=trn_corpus.shape[1],
                k=config['n_clusters'],
                gpu=config['gpu'],
                niter=config['max_iter'],
                nredo=config['nredo'],
                verbose=config['verbose'],
                seed=config['random_state']
            )
            
        except TypeError:
            raise Exception(
                f"clustering config {config} contains unexpected keyword arguments for CumlKMeans Clustering"
            )
            
        model.train(trn_corpus)
        
        # Get cluster assignments
        _, labels = model.index.search(trn_corpus, 1)
        labels = labels.flatten()
        
        # Create instance
        instance = cls(config, model)
        instance.centroids = model.centroids
        instance.labels_ = labels
        
        return instance

    def predict(self, predict_input):
        """Predict an input.

        Args:
            corpus (str, list): List of strings to predict.

        Returns:
            numpy.ndarray: Matrix of features.
        """

        if self.config.get('kwargs', {}).get('spherical', False):
            faiss.normalize_L2(predict_input)
            
        _, labels = self.model.index.search(predict_input, 1)
        return labels.flatten()

    def labels(self):
        """Get training labels.
        
        Returns:
            np.ndarray: Cluster labels from training.
        """
        return self.labels_
    
    
class BalancedKMeans(ClusteringModel):
    """Cuml KMeans with gpu support"""

    def __init__(self, config=None, model=None):
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
    def load(cls, load_dir, config):
        """Load a saved Cuml KMeans model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            CumlKMeans: The loaded object.
        """

        # LOGGER.info(f"Loading Cuml KMeans Clustering Model from {load_dir}")
        clustering_path = os.path.join(load_dir, "clustering.pkl")
        assert os.path.exists(
            clustering_path
        ), f"clustering path {clustering_path} does not exist"

        with open(clustering_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls(config, model_data)
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
            "n_clusters": 8,
            "distance": "cosine",
            "tol": 1e-4,
            "tqdm_flag": True,
            "iter_limit": 400,
            "iter_k": None,
            "device": None,
            "gamma_for_soft_dtw": 0.001,
        }

        try:
            config = {**defaults, **config}
            device = torch.device("cuda" if config["device"] == "gpu" and torch.cuda.is_available() else "cpu")
            model = PyTorchBalancedKMeans(n_clusters=config["n_clusters"], balanced=True, device=device)
        except TypeError:
            raise Exception(
                f"clustering config {config} contains unexpected keyword arguments for CumlKMeans Clustering"
            )

        # print(config["n_clusters"])
        # print(type(trn_corpus))
        trn_corpus = torch.from_numpy(trn_corpus)
        
        # Check for zero vectors (norm == 0)
        norms = torch.norm(trn_corpus, dim=1)
        if (norms == 0).any():
            # print(f"Warning: Found {torch.sum(norms == 0).item()} zero vectors in input!")
            # You might want to remove or fix these vectors, e.g.:
            trn_corpus = trn_corpus[norms > 0]
        
        cluster_labels = model.fit(X=trn_corpus, 
                                   distance=config["distance"], 
                                   tol=config["tol"], 
                                   tqdm_flag=config["tqdm_flag"],
                                   iter_limit=config["iter_limit"], 
                                   gamma_for_soft_dtw=config["gamma_for_soft_dtw"],
                                   iter_k=config["iter_k"]
                                   )
        cluster_labels = cluster_labels.cpu().numpy()
        model = {"cluster_labels": cluster_labels}
        return cls(config, model)

    def predict(self, predict_input):
        """Predict an input.

        Args:
            corpus (str, list): List of strings to predict.

        Returns:
            numpy.ndarray: Matrix of features.
        """

        return self.model.predict(predict_input)

    def labels(self):
        return self.model["cluster_labels"]
    
    # def centroids(self):
    #     return self.model["cluster_centers"]