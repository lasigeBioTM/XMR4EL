import importlib
import logging
import os
import json
import pickle
import pkgutil
import sys

import numpy as np

from abc import ABCMeta

from sklearn.linear_model import LogisticRegression

from src.models.dtree import DTree
from src.gpu_availability import is_cuda_available


linear_dict = {}

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if is_cuda_available():
    from cuml.linear_model import LogisticRegression as CUMLLogisticRegression
    

class LinearMeta(ABCMeta):
    """
    Metaclass for tracking all subclasses of ClusteringModel.
    Automatically registers each subclass in the cluster_dict.
    """
    def __new__(cls, name, bases, attr):
        new_cls = super().__new__(cls, name, bases, attr)
        if name != 'LinearModel':
            linear_dict[name.lower()] = new_cls
        return new_cls

    @classmethod
    def load_subclasses(cls, package_name):
        """Dynamically imports all modules in the package to register subclasses."""
        package = sys.modules[package_name]
        for _, modname, _ in pkgutil.iter_modules(package.__path__):
            importlib.import_module(f"{package_name}.{modname}")

class LinearModel(metaclass=LinearMeta):
    """Wrapper to all linear models"""
    def __init__(self, config, model):
        self.config = config
        self.model = model
    
    def save(self, linear_folder):
        """Save linear model to disk.

        Args:
            linear_folder (str): Folder to save to.
        """
        
        os.makedirs(linear_folder, exist_ok=True)
        with open(os.path.join(linear_folder, "linear_config.json"), "w", encoding="utf-8") as fout:
            fout.write(json.dumps(self.config))
        self.model.save(linear_folder)
    
    @classmethod
    def load(cls, linear_folder):
        """Load a saved linear model from disk.

        Args:
            linear_folder (str): Folder where `LinearModel` was saved to using `LinearModel.save`.

        Returns:
            LinearModel: The loaded object.
        """
        
        config_path = os.path.join(linear_folder, "linear_config.json")
        
        if not os.path.exists(config_path):
            config = {"type": "sklearnlogisticregression", 'kwargs': {}}
        else:
            with open(config_path, "r", encoding="utf-8") as fin:
                config = json.loads(fin.read())
                
        linear_type = config.get("type", None)
        assert linear_type is not None, f"{linear_folder} is not a valid linear folder"
        assert linear_type in linear_dict, f"invalid linear type {config['type']}"
        model = linear_dict[linear_type].load(linear_folder)
        return cls(config, model)
    
    @classmethod
    def train(cls, X_train, y_train=None, config=None, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list or str): Training corpus in the form of a list of strings or path to text file.
            config (dict, optional): Dict with key `"type"` and value being the lower-cased name of the specific cluster
            class to use.
                Also contains keyword arguments to pass to the specified cluster. Default behavior is to use logistic regression cluster with default arguments.
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            LinearModel: Trained linear model.
        """
        
        config = config if config is not None else {"type": "sklearnlogisticregression", "kwargs": {}}
        LOGGER.debug(f"Train Logistic Regression with config: {json.dumps(config, indent=True)}")
        linear_type = config.get("type", None)
        assert(
            linear_type is not None
        ), f"config {config} should contain a key 'type' for the linear type" 
        
        if y_train is None:
            assert isinstance(X_train, DTree), f"Trying to run an HierarchicalLinear but X_train is not of type({DTree.__class__})"
            model = linear_dict[linear_type].train(
            X_train, config=config["kwargs"], dtype=dtype
        )
        else: 
            model = linear_dict[linear_type].train(
                X_train, y_train, config=config["kwargs"], dtype=dtype
            )
        config['kwargs'] = model.config
        return cls(config, model)
    
    @staticmethod
    def load_config_from_args(args):
        """Parse config from a `argparse.Namespace` object.

        Args:
            args (argparse.Namespace): Contains either a `linear_config_path` (path to a json file) or `linear_config_json` (a json object in string form).

        Returns:
            dict: The dict resulting from loading the json file or json object.

        Raises:
            Exception: If json object cannot be loaded.
        """
        
        if args.linear_config_path is not None:
            with open(args.linear_config_path, "r", encoding="utf-8") as fin:
                linear_config_json = fin.read()
        else:
            linear_config_json = args.linear_config_json
        
        try:
            linear_config = json.loads(linear_config_json)
        except json.JSONDecodeError as jex:
            raise Exception(
                f"Failed to load clustering config json from {linear_config_json} ({jex})"
            )
        return linear_config
    
class SklearnLogisticRegression(LinearModel):
    """Sklearn Logistic Regression"""
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        
    def save(self, save_dir):
        """Save trained sklearn Logistic Regression model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "linear_model.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)
    
    @classmethod
    def load(cls, load_dir):
        """Load a saved sklearn Logistic Regression model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            SklearnLogisticRegression: The loaded object.
        """
        
        LOGGER.info(f"Loading Sklearn Logistic Regression linear model from {load_dir}")
        linear_path = os.path.join(load_dir, "linear_model.pkl")
        assert os.path.exists(linear_path), f"linear path {linear_path} does not exist"
        
        with open(linear_path, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model

    @classmethod
    def train(cls, X_train, y_train, config={}, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to sklearn's Logistic Regression.

        Returns:
            LogisticRegression: Trained Linear Model.

        Raises:
            Exception: If `config` contains keyword arguments that the SklearnLogisticRegression does not accept.
        """
        
        defaults = {
            'penalty': 'l2',
            'dual': False,
            'tol': 0.0001,
            'C': 1.0,
            'fit_intercept': True,
            'intercept_scaling': 1,
            'class_weight': None,
            'random_state': None,
            'solver': 'lbfgs',
            'max_iter': 100,
            'verbose': 0,
            'warm_start': False,
            'n_jobs': None,
            'l1_ratio': None
            }
        
        try:
            config = {**defaults, **config}
            model = LogisticRegression(**config)
        except TypeError:
            raise Exception(
                f"clustering config {config} contains unexpected keyword arguments for SklearnAgglomerativeClustering"
            )
        model.fit(X_train, y_train)
        return cls(config, model)
    
    def predict(self, predict_input):
        self.model.predict(predict_input)
        
class CumlLogisticRegression(LinearModel):
    """Cuml Logistic Regression"""
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        
    def save(self, save_dir):
        """Save trained sklearn Logistic Regression model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "linear_model.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)
    
    @classmethod
    def load(cls, load_dir):
        """Load a saved sklearn Logistic Regression model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            SklearnLogisticRegression: The loaded object.
        """
        
        LOGGER.info(f"Loading Cuml Logistic Regression linear model from {load_dir}")
        linear_path = os.path.join(load_dir, "linear_model.pkl")
        assert os.path.exists(linear_path), f"linear path {linear_path} does not exist"
        
        with open(linear_path, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model

    @classmethod
    def train(cls, X_train, y_train, config={}):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to sklearn's Logistic Regression.

        Returns:
            LogisticRegression: Trained Linear Model.

        Raises:
            Exception: If `config` contains keyword arguments that the CumlLogisticRegression does not accept.
        """
        
        defaults = {
            'penalty':'l2', 
            'tol': 0.0001,
            'C': 1.0, 
            'fit_intercept': True, 
            'class_weight': None, 
            'max_iter': 1000, 
            'linesearch_max_iter': 50,
            'verbose': False, 
            'l1_ratio': None, 
            'solver': 'qn', 
            'handle': None, 
            'output_type': None
            }
        
        try:
            config = {**defaults, **config}
            model = CUMLLogisticRegression(**config)
        except TypeError:
            raise Exception(
                f"clustering config {config} contains unexpected keyword arguments for SklearnAgglomerativeClustering"
            )
        model.fit(X_train, y_train)
        return cls(config, model)
    
    def predict(self, predict_input):
        self.model.predict(predict_input)