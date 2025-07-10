import json
import logging
import os
import pickle

import numpy as np

from abc import ABCMeta
from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer_dict = {}

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class VectorizerMeta(ABCMeta):
    """
    Metaclass for tracking all subclasses of Vectorizer.
    Automatically registers each subclass in the vectorizer_dict.
    """

    def __new__(cls, name, bases, attr):
        new_cls = super().__new__(cls, name, bases, attr)
        if name != "Vectorizer":
            vectorizer_dict[name.lower()] = new_cls
        return new_cls


class Vectorizer(metaclass=VectorizerMeta):
    """Wrapper class for all vectorizers."""

    def __init__(self, config, model):
        """Initialization

        Args:
            config (dict): Dict with key `"type"` and value being the lower-cased name of the specific vectorizer class to use.
                Also contains keyword arguments to pass to the specified vectorizer.
            model (Vectorizer): Trained vectorizer.
        """

        self.config = config
        self.model = model

    def save(self, vectorizer_folder):
        """Save trained vectorizer to disk.

        Args:
            vectorizer_folder (str): Folder to save to.
        """

        os.makedirs(vectorizer_folder, exist_ok=True)
        with open(
            os.path.join(vectorizer_folder, "vec_config.json"), "w", encoding="utf-8"
        ) as fout:
            fout.write(json.dumps(self.config))
        self.model.save(vectorizer_folder)

    @classmethod
    def load(cls, vectorizer_folder):
        """Load a saved vectorizer from disk.

        Args:
            vectorizer_folder (str): Folder where `Vectorizer` was saved to using `Vectorizer.save`.

        Returns:
            Vectorizer: The loaded object.
        """

        config_path = os.path.join(vectorizer_folder, "vec_config.json")

        if not os.path.exists(config_path):
            config = {"type": "tfidf", "kwargs": {}}
        else:
            with open(config_path, "r", encoding="utf-8") as fin:
                config = json.loads(fin.read())

        vectorizer_type = config.get("type", None)
        assert (
            vectorizer_type is not None
        ), f"{vectorizer_folder} is not a valid vectorizer folder"
        assert (
            vectorizer_type in vectorizer_dict
        ), f"invalid vectorizer type {config['type']}"
        model = vectorizer_dict[vectorizer_type].load(vectorizer_folder)
        return cls(config, model)

    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list or str): Training corpus in the form of a list of strings or path to text file.
            config (dict, optional): Dict with key `"type"` and value being the lower-cased name of the specific vectorizer class to use.
                Also contains keyword arguments to pass to the specified vectorizer. Default behavior is to use tfidf vectorizer with default arguments.
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            Vectorizer: Trained vectorizer.
        """

        config = config if config is not None else {"type": "tfidf", "kwargs": {}}
        LOGGER.debug(f"Train Vectorizer with config: {json.dumps(config, indent=True)}")
        vectorizer_type = config.get("type", None)
        assert (
            vectorizer_type is not None
        ), f"config {config} should contain a key 'type' for the vectorizer type"
        model = vectorizer_dict[vectorizer_type].train(
            trn_corpus, config=config["kwargs"], dtype=dtype
        )
        config["kwargs"] = model.config
        return cls(config, model)

    def predict(self, corpus, **kwargs):
        """Vectorize a corpus.

        Args:
            corpus (list or str): List of strings to vectorize or path to text file.
            **kwargs: Keyword arguments to pass to the trained vectorizer.

        Returns:
            numpy.ndarray or scipy.sparse.csr.csr_matrix: Matrix of features.
        """

        if isinstance(corpus, str) and self.config["type"] != "tfidf":
            raise ValueError(
                "Iterable over raw text expected for vectorizer other than tfidf."
            )
        return self.model.predict(corpus, **kwargs)

    @staticmethod
    def load_config_from_args(args):
        """Parse config from a `argparse.Namespace` object.

        Args:
            args (argparse.Namespace): Contains either a `vectorizer_config_path` (path to a json file) or `vectorizer_config_json` (a json object in string form).

        Returns:
            dict: The dict resulting from loading the json file or json object.

        Raises:
            Exception: If json object cannot be loaded.
        """

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


class Tfidf(Vectorizer):
    """
    Sklearn tfidf vectorizer
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    """

    def __init__(self, config=None, model=None):
        """Initialization

        Args:
            config (dict): Dict with key `"type"` and value being the lower-cased name of the specific vectorizer class to use.
                Also contains keyword arguments to pass to the specified vectorizer.
            model (sklearn.feature_extraction.text.TfidfVectorizer, optional): The trained tfidf vectorizer. Default is `None`.
        """

        self.config = config
        self.model = model

    def __del__(self):
        """Destruct self model instance"""
        self.model = None

    def save(self, save_dir):
        """Save trained sklearn Tfidf vectorizer to disk.

        Args:
            save_dir (str): Folder to store serialized object in.
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "vectorizer.pkl"), "wb") as fout:
            pickle.dump(self.__dict__, fout)

    @classmethod
    def load(cls, load_dir):
        """Load a saved sklearn Tfidf vectorizer from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            Tfidf: The loaded object.
        """

        vectorizer_path = os.path.join(load_dir, "vectorizer.pkl")
        assert os.path.exists(
            vectorizer_path
        ), f"vectorizer path {vectorizer_path} does not exist"

        with open(vectorizer_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model

    @classmethod
    def train(cls, trn_corpus, config={}, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to sklearn's TfidfVectorizer.
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            Tfidf: Trained vectorizer.

        Raises:
            Exception: If `config` contains keyword arguments that the tfidf vectorizer does not accept.
        """
        defaults = {
            "ngram_range": (1, 2),  # n-grams from 1 to 2
            "max_features": 50000,  # No max feature limit
            "min_df": 0.0,  # Minimum document frequency ratio
            "max_df": 0.98,  # Maximum document frequency ratio
            "binary": False,  # Term frequency is not binary
            "use_idf": True,  # Use inverse document frequency
            "smooth_idf": True,  # Apply smoothing to idf
            "sublinear_tf": True,  # Use raw term frequency
            "norm": "l2",  # Apply L2 normalization
            "analyzer": "word",  # Tokenizes by word
            "stop_words": "english",  # No stop words used
        }

        try:
            config = {**defaults, **config}
            model = TfidfVectorizer(**config, dtype=dtype)
        except TypeError:
            raise Exception(
                f"vectorizer config {config} contains unexpected keyword arguments for TfidfVectorizer"
            )
        model.fit(trn_corpus)
        return cls(config, model)

    def predict(self, corpus):
        """Vectorize a corpus.

        Args:
            corpus (list): List of strings to vectorize.

        Returns:
            scipy.sparse.csr.csr_matrix: Matrix of features.
        """
        return self.model.transform(corpus)
