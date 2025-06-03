import importlib
import logging
import os
import json
import pickle
import pkgutil
import sys
import torch

import numpy as np

from abc import ABCMeta

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SVC


classifier_dict = {}

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if torch.cuda.is_available():
    from cuml.linear_model import LogisticRegression as CUMLLogisticRegression


class ClassifierMeta(ABCMeta):
    """
    Metaclass for tracking all subclasses of ClusteringModel.
    Automatically registers each subclass in the cluster_dict.
    """

    def __new__(cls, name, bases, attr):
        new_cls = super().__new__(cls, name, bases, attr)
        if name != "ClassifierModel":
            classifier_dict[name.lower()] = new_cls
        return new_cls

    @classmethod
    def load_subclasses(cls, package_name):
        """Dynamically imports all modules in the package to register subclasses."""

        package = sys.modules[package_name]
        for _, modname, _ in pkgutil.iter_modules(package.__path__):
            importlib.import_module(f"{package_name}.{modname}")


class ClassifierModel(metaclass=ClassifierMeta):
    """Wrapper to all classifier models"""

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def save(self, classifier_folder):
        """Save classifier model to disk.

        Args:
            classifier_folder (str): Folder to save to.
        """

        os.makedirs(classifier_folder, exist_ok=True)
        with open(
            os.path.join(classifier_folder, "classifier_config.json"),
            "w",
            encoding="utf-8",
        ) as fout:
            fout.write(json.dumps(self.config))
        self.model.save(classifier_folder)

    @classmethod
    def load(cls, classifier_folder):
        """Load a saved classifier model from disk.

        Args:
            classifier_folder (str): Folder where `classifierModel` was saved to using `classifierModel.save`.

        Returns:
            classifierModel: The loaded object.
        """

        config_path = os.path.join(classifier_folder, "classifier_config.json")

        if not os.path.exists(config_path):
            config = {"type": "sklearnlogisticregression", "kwargs": {}}
        else:
            with open(config_path, "r", encoding="utf-8") as fin:
                config = json.loads(fin.read())

        classifier_type = config.get("type", None)
        assert (
            classifier_type is not None
        ), f"{classifier_folder} is not a valid classifier folder"
        assert (
            classifier_type in classifier_dict
        ), f"invalid classifier type {config['type']}"
        model = classifier_dict[classifier_type].load(classifier_folder, config)
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
            classifierModel: Trained classifier model.
        """

        config = (
            config
            if config is not None
            else {"type": "sklearnlogisticregression", "kwargs": {}}
        )
        LOGGER.debug(
            f"Training classifier with config: {json.dumps(config, indent=True)}"
        )
        classifier_type = config.get("type", None)
        assert (
            classifier_type is not None
        ), f"config {config} should contain a key 'type' for the classifier type"

        model = classifier_dict[classifier_type].train(
            X_train, y_train, config=config["kwargs"], dtype=dtype
        )

        config["kwargs"] = model.config
        return cls(config, model)

    def predict(self, predict_input):
        return self.model.predict(predict_input)

    def predict_proba(self, predict_input):
        return self.model.predict_proba(predict_input)

    @staticmethod
    def load_config_from_args(args):
        """Parse config from a `argparse.Namespace` object.

        Args:
            args (argparse.Namespace): Contains either a `classifier_config_path` (path to a json file) or `classifier_config_json` (a json object in string form).

        Returns:
            dict: The dict resulting from loading the json file or json object.

        Raises:
            Exception: If json object cannot be loaded.
        """

        if args.classifier_config_path is not None:
            with open(args.classifier_config_path, "r", encoding="utf-8") as fin:
                classifier_config_json = fin.read()
        else:
            classifier_config_json = args.classifier_config_json

        try:
            classifier_config = json.loads(classifier_config_json)
        except json.JSONDecodeError as jex:
            raise Exception(
                f"Failed to load clustering config json from {classifier_config_json} ({jex})"
            )
        return classifier_config


class SklearnLogisticRegression(ClassifierModel):
    """Sklearn Logistic Regression"""

    def __init__(self, config=None, model=None):
        self.config = config
        self.model = model

    def save(self, save_dir):
        """Save trained sklearn Logistic Regression model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "classifier_model.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir, config):
        """Load a saved sklearn Logistic Regression model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            SklearnLogisticRegression: The loaded object.
        """

        LOGGER.info(
            f"Loading Sklearn Logistic Regression Classifier Model from {load_dir}"
        )
        classifier_path = os.path.join(load_dir, "classifier_model.pkl")
        assert os.path.exists(
            classifier_path
        ), f"Classifier path {classifier_path} does not exist"

        with open(classifier_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls(config, model_data)
        return model

    @classmethod
    def train(cls, X_train, y_train, config={}, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to sklearn's Logistic Regression.

        Returns:
            LogisticRegression: Trained classifier Model.

        Raises:
            Exception: If `config` contains keyword arguments that the SklearnLogisticRegression does not accept.
        """

        defaults = {
            "penalty": "l2",
            "dual": False,
            "tol": 0.0001,
            "C": 1.0,
            "fit_intercept": True,
            "intercept_scaling": 1,
            "class_weight": None,
            "random_state": None,
            "solver": "lbfgs",
            "max_iter": 100,
            "verbose": 0,
            "warm_start": False,
            "n_jobs": None,
            "l1_ratio": None,
        }

        try:
            config = {**defaults, **config}
            model = LogisticRegression(**config)
        except TypeError:
            raise Exception(
                f"clustering config {config} contains unexpected keyword arguments for SklearnLogisticRegression"
            )
        model.fit(X_train, y_train)
        return cls(config, model)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class SklearnRandomForestClassifier(ClassifierModel):
    """SKlearn Random Forest"""

    def __init__(self, config=None, model=None):
        self.config = config
        self.model = model

    def save(self, save_dir):
        """Save trained sklearn Logistic Regression model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "classifier_model.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir, config):
        """Load a saved sklearn Logistic Regression model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            SklearnLogisticRegression: The loaded object.
        """

        LOGGER.info(f"Loading Sklearn Random Forest Classifier Model from {load_dir}")
        classifier_path = os.path.join(load_dir, "classifier_model.pkl")
        assert os.path.exists(
            classifier_path
        ), f"Classifier path {classifier_path} does not exist"

        with open(classifier_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls(config, model_data)
        return model

    @classmethod
    def train(cls, X_train, y_train, config={}, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to sklearn's Logistic Regression.

        Returns:
            LogisticRegression: Trained classifier Model.

        Raises:
            Exception: If `config` contains keyword arguments that the SklearnLogisticRegression does not accept.
        """

        defaults = {
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": "sqrt",
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "bootstrap": True,
            "oob_score": False,
            "n_jobs": None,
            "random_state": None,
            "verbose": 0,
            "warm_start": False,
            "class_weight": None,
            "ccp_alpha": 0.0,
            "max_samples": None,
            "monotonic_cst": None,
        }

        try:
            config = {**defaults, **config}
            model = RandomForestClassifier(**config)
        except TypeError:
            raise Exception(
                f"clustering config {config} contains unexpected keyword arguments for SklearnRandomForest"
            )
        model.fit(X_train, y_train)
        return cls(config, model)

    def predict(self, predict_input):
        return self.model.predict(predict_input)


class SklearnSupportVectorClassification(ClassifierModel):
    """SKlearn SVC"""

    def __init__(self, config=None, model=None):
        self.config = config
        self.model = model

    def save(self, save_dir):
        """Save trained sklearn Support Vector Classifier model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "classifier_model.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir, config):
        """Load a saved sklearn Support Vector Classifier from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            SklearnLogisticRegression: The loaded object.
        """

        LOGGER.info(f"Loading Sklearn Support Vector Classifier Model from {load_dir}")
        classifier_path = os.path.join(load_dir, "classifier_model.pkl")
        assert os.path.exists(
            classifier_path
        ), f"Classifier path {classifier_path} does not exist"

        with open(classifier_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls(config, model_data)
        return model

    @classmethod
    def train(cls, X_train, y_train, config={}, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to sklearn's Logistic Regression.

        Returns:
            LogisticRegression: Trained classifier Model.

        Raises:
            Exception: If `config` contains keyword arguments that the SklearnLogisticRegression does not accept.
        """

        defaults = {
            "C": 1.0, 
            "kernel": 'rbf', 
            "degree": 3, 
            "gamma": 'scale', 
            "coef0": 0.0, 
            "shrinking": True, 
            "probability": True, 
            "tol": 0.001, 
            "cache_size": 200,
            "class_weight": None, 
            "verbose": False, 
            "max_iter": -1, 
            "decision_function_shape": 'ovr', 
            "break_ties": False, 
            "random_state": None
        }

        try:
            config = {**defaults, **config}
            model = SVC(**config)
        except TypeError:
            raise Exception(
                f"clustering config {config} contains unexpected keyword arguments for SklearnSVC"
            )
        model.fit(X_train, y_train)
        return cls(config, model)

    def predict(self, predict_input):
        return self.model.predict(predict_input)

class CumlLogisticRegression(ClassifierModel):
    """Cuml Logistic Regression"""

    def __init__(self, config=None, model=None):
        self.config = config
        self.model = model

    def save(self, save_dir):
        """Save trained sklearn Logistic Regression model to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "classifier_model.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, load_dir, config):
        """Load a saved sklearn Logistic Regression model from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            SklearnLogisticRegression: The loaded object.
        """

        LOGGER.info(
            f"Loading Cuml Logistic Regression Classifier Model from {load_dir}"
        )
        classifier_path = os.path.join(load_dir, "classifier_model.pkl")
        assert os.path.exists(
            classifier_path
        ), f"Classifier path {classifier_path} does not exist"

        with open(classifier_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls(config, model_data)
        return model

    @classmethod
    def train(cls, X_train, y_train, config={}):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to sklearn's Logistic Regression.

        Returns:
            LogisticRegression: Trained classifier Model.

        Raises:
            Exception: If `config` contains keyword arguments that the CumlLogisticRegression does not accept.
        """

        defaults = {
            "penalty": "l2",
            "tol": 0.0001,
            "C": 1.0,
            "fit_intercept": True,
            "class_weight": None,
            "max_iter": 1000,
            "linesearch_max_iter": 50,
            "verbose": False,
            "l1_ratio": None,
            "solver": "qn",
            "handle": None,
            "output_type": None,
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
        return self.model.predict(predict_input)
    
class DeepMLPClassifier(ClassifierModel):

    def __init__(self, config, model):
        self.config = config
        self.model = model
        
    def save(self, save_dir):
        pass
    
    @classmethod
    def load(cls, load_dir, config):
        pass
    
    @classmethod
    def train(cls, X_train, y_train, config={}):
        pass
    
    
    
