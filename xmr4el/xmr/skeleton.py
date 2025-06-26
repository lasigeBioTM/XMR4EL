import os
import pickle
import logging
import glob
import json

import numpy as np

from datetime import datetime
from collections import Counter

from xmr4el.featurization.vectorizers import Vectorizer
from xmr4el.models.classifier_wrapper.classifier_model import ClassifierModel
from xmr4el.models.cluster_wrapper.clustering_model import ClusteringModel

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Skeleton:
    """
    A hierarchical tree structure for Extreme Multi-label Ranking (XMR) that stores:
    - Embeddings at each level (text, transformer, concatenated)
    - Trained models (vectorizer, clustering, classifier)
    - Label information and PIFA embeddings
    - Child nodes for hierarchical structure
    """
    
    def __init__(
        self,
        train_data=None,
        pifa_embeddings=None,
        text_embeddings=None,
        transformer_embeddings=None,
        concatenated_embeddings=None,
        kb_indices=None,
        vectorizer=None,
        clustering_model=None,
        classifier_model=None,
        test_split=None,
        reranker=None,
        children=None,  # Dictionary of XMRTree nodes
        depth=0,
    ):
        """
        Initialize an XMRTree node with optional components.
        
        Args:
            train_data: Dictionary, label: corpus
            pifa_embeddings: PIFA label embeddings for this node
            text_embeddings: Text (e.g., TF-IDF) embeddings
            transformer_embeddings: Transformer model embeddings
            concatenated_embeddings: Combined feature embeddings
            kb_indices: Indices mapping to knowledge base entries
            vectorizer: Text vectorizer model
            clustering_model: Clustering model for this node
            classifier_model: Classifier model for this node
            test_split: Train/test split data for evaluation
            children: Dictionary of child nodes (key=cluster ID, value=XMRTree)
            depth: Current depth in the hierarchy (0=root)
        """
        # Training data
        self.train_data = train_data
        
        # Label information
        self.pifa_embeddings = pifa_embeddings

        # Embeddings storage
        self.text_embeddings = text_embeddings
        self.transformer_embeddings = transformer_embeddings
        self.concatenated_embeddings = concatenated_embeddings
        self.kb_indices = kb_indices # Indices of data points in this node
        
        # Models
        self.vectorizer = vectorizer
        self.clustering_model = clustering_model
        self.classifier_model = classifier_model
        self.test_split = test_split # Evaluation Data
        self.reranker = reranker

        # Tree Structure
        self.children = children if children is not None else {} # Child Nodes
        self.depth = depth # Depth in hierarchy

    def save(self, save_dir="data/saved_trees", save_name=None, child_tree=False, timestamp=True):
        """
        Save trained XMRTree model to disk with proper organization.
        
        Args:
            save_dir (str): Base directory for saving
            child_tree (bool): If True, skips timestamp directory creation
                             (used when saving child trees recursively)
        """
        if not child_tree:
            if save_name is None:
                # For main tree, create timestamped directory
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_dir = os.path.join(save_dir, f"{self.__class__.__name__}_{timestamp}")
            else:
                save_dir = os.path.join(save_dir, f"{save_name}")
                
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

        # Prepare state dictionary (excluding large objects saved separately)
        state = self.__dict__.copy()

        # Save models individually (vectorizer, clustering, classifier)
        models = ["vectorizer", "clustering_model", "classifier_model"]
        models_data = [getattr(self, model_name, None) for model_name in models]

        for idx, model in enumerate(models_data):
            if model is not None:  # Ensure model exists before saving
                print(type(model))
                model.save(os.path.join(save_dir, models[idx]))

        # Remove models from state to avoid duplicate saving
        for model in models:
            state.pop(model, None)
        
        train_data = getattr(self, "train_data", None)    
        
        if not child_tree and train_data:
            train_data_path = os.path.join(save_dir, "train_dict.json")
            with open(train_data_path, 'w', encoding='utf-8') as train_f:
                json.dump(self.train_data, train_f, indent=4)
            state.pop("train_data", None)
            LOGGER.info(f"Loaded Training data, {train_data_path}")

        # Save large embeddings as numpy files
        for attr in [
            "text_embeddings",
            "transformer_embeddings",
            "concatenated_embeddings",
        ]:
            emb_data = getattr(self, attr, None)
            if emb_data is not None:
                np.save(os.path.join(save_dir, f"{attr}.npy"), emb_data)
                state.pop(attr, None) # Remove from state dict
            else:
                LOGGER.warning(f"{attr} is None and will not be saved.")

        # Handle child trees recursively
        state.pop("children", {}) # Children saved separately
        if self.children:
            children_dir = os.path.join(save_dir, "children")
            os.makedirs(children_dir, exist_ok=True)
            for idx, child in self.children.items():
                child_save_dir = os.path.join(
                    children_dir, f"child_{idx}"
                )  # Unique directory for each child
                child.save(child_save_dir, child_tree=True)

        # Save remaining metadata as pickle
        with open(os.path.join(save_dir, "xmrtree.pkl"), "wb") as fout:
            pickle.dump(state, fout)

        LOGGER.info(f"Model saved successfully at {save_dir}")

    @classmethod
    def load(cls, load_dir=None, child_tree=False):
        """
        Load a saved XMRTree from disk.
        
        Args:
            load_dir (str): Directory to load from. If None, loads most recent.
            child_tree (bool): If True, indicates loading a child tree.
            
        Returns:
            XMRTree: The reconstructed tree structure
        """
        if load_dir is None:
            # Find most recent save if no directory specified
            all_saves = sorted(
                glob.glob(os.path.join("data/saved_trees", f"{cls.__name__}_*")),
                reverse=True,
            )
            if not all_saves:
                raise FileNotFoundError(
                    f"No saved {cls.__name__} models found in saved_trees."
                )
            load_dir = all_saves[0]  # Load the most recent one
            LOGGER.info(f"Loading latest model from: {load_dir}")
            
        # Load main metadata
        tree_path = os.path.join(load_dir, "xmrtree.pkl")
        assert os.path.exists(tree_path), f"XMRTree path {tree_path} does not exist"

        with open(tree_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)
        
        train_data_path_json = os.path.join(load_dir, "train_dict.json")

        if os.path.exists(train_data_path_json):
            with open(train_data_path_json, 'r', encoding='utf-8') as train_f:
                model.train_data = json.load(train_f)
        else:
            model.train_data = None

        # Load models (skip vectorizer for child trees)
        if not child_tree:
            setattr(
                model,
                "vectorizer",
                Vectorizer.load(os.path.join(load_dir, "vectorizer")),
            )

        setattr(
            model,
            "clustering_model",
            ClusteringModel.load(os.path.join(load_dir, "clustering_model")),
        )
        setattr(
            model,
            "classifier_model",
            ClassifierModel.load(os.path.join(load_dir, "classifier_model")),
        )

        # Load embeddings if they exist
        for attr in [
            "text_embeddings",
            "transformer_embeddings",
            "concatenated_embeddings",
        ]:
            emb_path = os.path.join(load_dir, f"{attr}.npy")
            if os.path.exists(emb_path):
                setattr(model, attr, np.load(emb_path, allow_pickle=True))
                LOGGER.debug(f"Loaded {attr} embeddings from {emb_path}")
            else:
                pass
                LOGGER.debug(f"{attr} not found at {emb_path}")

        # Recursively load child trees
        children_dir = os.path.join(load_dir, "children")
        if os.path.exists(children_dir):
            for child_name in os.listdir(children_dir):
                child_tree_path = os.path.join(children_dir, child_name)
                if os.path.isdir(child_tree_path):  # Make sure it's a directory
                    try:
                        child_index = int(child_name.replace("child_", ""))  # Convert 'child_0' -> 0
                        child_model = cls.load(child_tree_path, child_tree=True)  # Recursively load child
                        model.children[child_index] = child_model
                        LOGGER.debug(
                        f"Loading child tree from {child_tree_path} as index {child_index}"
                        )
                    except ValueError:
                        LOGGER.debug(f"Skipping unexpected child folder: {child_name}")

        LOGGER.info(f"Model loaded successfully from {load_dir}")
        return model
    
    def set_train_data(self, train_data):
        """Set Training Data"""
        self.train_data = train_data
        
    def set_pifa_embeddings(self, pifa_embeddings):
        """Set PIFA label embeddings for this node."""
        self.pifa_embeddings = pifa_embeddings

    def set_text_embeddings(self, text_embeddings):
        """Set text embeddings (e.g., TF-IDF) for this node."""
        self.text_embeddings = text_embeddings

    def set_transformer_embeddings(self, transformer_embeddings):
        """Set transformer model embeddings for this node."""
        self.transformer_embeddings = transformer_embeddings

    def set_concatenated_embeddings(self, concatenated_embeddings):
        """Set concatenated feature embeddings for this node."""
        self.concatenated_embeddings = concatenated_embeddings
        
    def set_kb_indices(self, kb_indices):
        """Set knowledge base indices for this node's data points."""
        self.kb_indices = kb_indices

    def set_vectorizer(self, vectorizer):
        """Set the text vectorizer model (typically only at root)."""
        self.vectorizer = vectorizer

    def set_clustering_model(self, clustering_model):
        """Set the clustering model for this node."""
        self.clustering_model = clustering_model

    def set_classifier_model(self, classifier_model):
        """Set the classifier model for this node."""
        self.classifier_model = classifier_model

    def set_test_split(self, test_split):
        """Set train/test split data for evaluation."""
        self.test_split = test_split
        
    def set_reranker(self, reranker):
        """Set Reranker for evaluation"""
        self.reranker = reranker

    def set_children(self, idx, child_tree):
        """Add a child node at the specified index."""
        self.children[idx] = child_tree

    def set_depth(self, depth):
        """Set the depth of this node in the hierarchy."""
        self.depth = depth

    def is_empty(self):
        """Check if this node is empty (no clustering model)."""
        return self.clustering_model is None

    def is_leaf(self):
        """Check if this node is a leaf (no children)."""
        return self.children is None        

    def __str__(self, indent="", last=True, max_depth=None):
        """
        String representation showing hierarchical tree structure with detailed cluster information.
        
        Args:
            indent (str): Current indentation string (used internally for recursion)
            last (bool): Whether this node is the last child of its parent
            max_depth (int): Maximum depth to print (None for unlimited)
            
        Returns:
            str: Formatted tree structure string with cluster details
        """
        if max_depth is not None and self.depth > max_depth:
            return ""
            
        # Tree structure components
        prefix = "└── " if last else "├── "
        new_indent = indent + ("    " if last else "│   ")
        
        # Node information
        node_info = f"Depth {self.depth}"
        
        # Cluster statistics
        stats = []
        # if self.train_data:
        #     stats.append(f"Labels: {len(self.train_data)}")
        if self.kb_indices is not None:
            stats.append(f"KB Indices: {len(self.kb_indices)}")

        # Detailed cluster information
        cluster_info = ""
        if self.clustering_model and hasattr(self.clustering_model.model, 'labels_'):
            label_counts = Counter(self.clustering_model.model.labels_)
            stats.append(f"Clusters: {len(label_counts)}")
            
            # Add detailed cluster distribution (sorted by size)
            sorted_clusters = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))
            cluster_dist = ", ".join(f"C{i}:{count}" for i, (_, count) in enumerate(sorted_clusters))
            cluster_info = f"\n{new_indent}    Cluster distribution: {cluster_dist}"
        
        # Combine all information
        result = indent + prefix + node_info
        if stats:
            result += " [" + ", ".join(stats) + "]"
        result += cluster_info + "\n"
        
        # Recursively add children
        if self.children:
            child_count = len(self.children)
            for i, (key, child) in enumerate(self.children.items()):
                result += child.__str__(
                    new_indent,
                    i == child_count - 1,
                    max_depth
                )
        
        return result

    def __repr__(self):
        """Compact representation for debugging."""
        stats = []
        if self.train_data:
            stats.append(f"labels={len(self.train_data)}")
        if self.clustering_model and hasattr(self.clustering_model, 'labels_'):
            label_counts = Counter(self.clustering_model.labels_)
            stats.append(f"clusters={len(label_counts)}")
        if self.kb_indices is not None:
            stats.append(f"points={len(self.kb_indices)}")
        
        return f"Skeleton(depth={self.depth}, children={len(self.children)}, {', '.join(stats)})"