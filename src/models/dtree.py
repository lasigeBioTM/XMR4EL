import numpy as np

from typing import Dict, Optional
from collections import Counter

from src.models.cluster_wrapper.clustering_model import ClusteringModel


class DTree():
    
    def __init__(self, node=None, depth=0, parent=None):
        self.depth = depth
        self.parent = parent
        self.children = {}
        
        if node is None:
            self.node = Node()
        else:
            self.node = node
        
    def is_leaf(self):
        return len(self.children) == 0    
    
    def set_child_dtree(self, cluster_label, child):
        # print(f"\n{self.children}")
        assert cluster_label not in self.children, f"Duplicated value of cluster label {cluster_label} at depth {self.depth}"
        assert isinstance(child, DTree), f"Expected child to be DTree, got {type(child).__name__}"
        
        child.parent = self
        self.children[cluster_label] = child
        
    def __str__(self):
        # Start with the current node and its depth
        indent = f"{' ' * (self.depth + 1) * 5}"
        result = f"{indent}Node at depth {self.depth} - "
        
        if self.is_leaf():
            result += "Leaf Node"
        else:
            result += "Non-Leaf Node"
        
        # Add the node type (ClusterNode or LinearNode) for detailed understanding
        if isinstance(self.node.cluster_node, ClusterNode):
            unique_labels = len(set(self.node.cluster_node.labels)) if self.node.cluster_node.labels is not None else 0
            result += f" - ClusterNode with {unique_labels} unique labels \n{indent} - Labels count\n"
            for idx, item in sorted(Counter(self.node.cluster_node.labels).items()):
                result += f"{indent}    {idx} -> {item}\n"
        elif isinstance(self.node.linear_node, LinearNode):
            result += f" - LinearNode with model: {self.node.model}"

        # Recursively print children if the node is not a leaf
        if not self.is_leaf():
            result += "\n"
            for label, child in self.children.items():
                result += f"{'  ' * (self.depth + 1) * 5}Child {label}: " + str(child) + "\n"

        return result.strip()
        
class Node():
    
    def __init__(self, cluster_node=None, linear_node=None):
        self.cluster_node = cluster_node
        self.linear_node = linear_node
        
    def set_cluster_node(self, model, cluster_points, config):
        self.cluster_node = ClusterNode(model, cluster_points, config)
        
    def set_linear_node(self, model, test_split):
        """ Assigns a classification model to this node. """
        self.linear_node = LinearNode(model=model, test_split=test_split)
        
class ClusterNode():
    
    def __init__(self, model=None, cluster_points=None, config=None):
        self.model: ClusteringModel = model
        self.cluster_points: Optional[np.ndarray] = cluster_points
        self.config = config
        
        if cluster_points is not None:
            self.labels = model.model.labels_
        else:
            self.labels = None
    
    def is_populated(self):
        return self.model is not None
    
class LinearNode():
    
    def __init__(self, model, test_split):
        self.model = model
        self.test_split: Dict[str, list] = test_split
    

    
    