import numpy as np

from typing import Dict, Optional

from src.models.cluster_wrapper.clustering_model import ClusteringModel


class DTree():
    
    def __init__(self, node=None, depth=-1, parent=None, children={}):
        self.depth = depth
        self.parent = parent
        self.children = children
        
        if node is None:
            self.node = Node()
        else:
            self.node = node
        
    def is_leaf(self):
        return len(self.children) == 0    
    
    def set_child_dtree(self, cluster_label, child):
        assert cluster_label not in self.children, f"Duplicated value of cluster label"
        assert isinstance(child, DTree), f"Expected child to be DTree, got {type(child).__name__}"
        
        child.parent = self
        self.children[cluster_label] = child
    
    def __str__(self, indent=0):
        return str(self.depth)
    
    def __str__(self) -> str:
        """ Returns a string representation of the TreeNode. """
        indent = "  " * self.depth * 2
        info = f"{indent}Depth: {self.depth}\n" if self.node.cluster_node else ""

        if self.node.cluster_node:
            if len(self.children.values()) == 0:
                info += f"{indent}Cluster Info: {self.node.cluster_node}, No Child Clusters\n"
            else:
                n_cluster = ""
                for child_cluster in self.children.keys():
                    n_cluster += str(child_cluster) + " "
                    
                info += f"{indent}Cluster Info: {self.node.cluster_node}, Child Clusters: {n_cluster}\n"

        if self.node.linear_node:
            info += f"{indent}Classifier Info: {self.node.linear_node.__str__(indent)}\n"

        for child in self.children.values():
            info += str(child)  # Recursively print children

        return info
        
class Node():
    
    def __init__(self, cluster_node=None, linear_node=None):
        self.cluster_node = cluster_node
        self.linear_node = linear_node
        
    def set_cluster_node(self, model, cluster_points, config):
        self.cluster_node = ClusterNode(model, cluster_points, config)
        
    def set_linear_node(self, model, test_split):
        """ Assigns a classification model to this node. """
        self.linear_node = LinearNode(model=model, test_split=test_split)
        
    def __str__(self):
        node_str = ""
        if self.cluster_node:
            node_str += f"ClusterNode: {str(self.cluster_node)}\n"
        if self.linear_node:
            node_str += f"LinearNode: {str(self.linear_node)}\n"
        return node_str
        
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

    def __str__(self):
        return f"Model: {self.model.__class__.__name__}, Labels: {self.labels if self.labels is not None else 'None'}"
    
class LinearNode():
    
    def __init__(self, model, test_split):
        self.model = model
        self.test_split: Dict[str, list] = test_split
    
    def __str__(self):
        return f"Model: {self.model.__class__.__name__}, Test Split: {self.test_split}"

    
    