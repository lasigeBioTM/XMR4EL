import numpy as np
from typing import Optional, Dict, Any


class ClusterNode:
    """ Represents a clustering node within the hierarchical tree. """
        
    def __init__(self, model: Optional[Any] = None, 
                 labels: Optional[np.ndarray] = None, 
                 cluster_points: Optional[np.ndarray] = None):
        
        self.model = model
        self.labels = labels
        self.cluster_points = cluster_points
        self.overall_silhouette_score: Optional[float] = None
        self.silhouette_scores: Dict[int, float] = {}
        
    def set_silhouette_scores(self, overall_score: float, score_dict: Dict[int, float]):
        """ Sets silhouette scores for the node. """
        self.overall_silhouette_score = overall_score
        self.silhouette_scores = score_dict

    def __str__(self, indent=None) -> str:
        """ Returns a string representation of the ClusterNode. """
        num_clusters = len(set(self.labels)) if self.labels is not None else 0
        unique_labels = np.unique(self.labels) if self.labels is not None else []
        
        silhouette_scores = ""
        if self.silhouette_scores:
            for idx, score in self.silhouette_scores.items():
                silhouette_scores += f"{int(idx)}: {float(score)}\n"
        
        # , \nOverall Silhouette Score: {self.overall_silhouette_score}, \nSilhouette Scores:\n{silhouette_scores}
        return f"Model: {type(self.model).__name__}, Clusters: {num_clusters}, Unique Labels: {unique_labels}"

class LinearNode:
    """ Represents a classification model at a specific tree node. """
    
    def __init__(self, model: Optional[Any] = None, 
                 top_k_score: Optional[float] = None, 
                 X_test: Optional[np.ndarray] = None, 
                 y_test: Optional[np.ndarray] = None):
        
        self.model = model
        self.top_k_score = top_k_score
        self.X_test = X_test
        self.y_test = y_test
        
    def __str__(self, indent=None) -> str:
        """ Returns a string representation of the LinearNode. """
        shape_info = f"X_test Shape: {self.X_test.shape}" if self.X_test is not None else "No test data"
        return f"Linear Model: {type(self.model).__name__}\n{indent}Top-K Score: {self.top_k_score}\n{indent}{shape_info}\n"
        
    
class TreeNode:
    """ Represents a hierarchical node in the classification tree. """
    
    def __init__(self, depth: int = 0, parent: Optional['TreeNode'] = None):
        self.depth = depth
        self.cluster_node: Optional[ClusterNode] = None
        self.linear_node: Optional[LinearNode] = None
        self.parent = parent
        self.children: Dict[int, 'TreeNode'] = {}
        
    def is_leaf(self) -> bool:
        """ Returns True if the node has no children, meaning it's a leaf. """
        return len(self.children) == 0
    
    def add_child(self, cluster_label: int, child_node: "TreeNode"):
        """ Adds a child node under the current tree node. """
        if cluster_label in self.children:
            raise ValueError(f"Cluster label {cluster_label} already exists as a child.")
        
        child_node.parent = self
        child_node.parent_cluster_label = cluster_label
        self.children[cluster_label] = child_node
    
    def set_cluster_node(self, clustering_model: Any, cluster_labels: np.ndarray, cluster_points: np.ndarray):
        """ Assigns a clustering model to this node. """
        self.cluster_node = ClusterNode(model=clustering_model, labels=cluster_labels, cluster_points=cluster_points)

    def set_linear_node(self, linear_model: Any, top_k_score: float, X_test: np.ndarray, y_test: np.ndarray):
        """ Assigns a classification model to this node. """
        self.linear_node = LinearNode(model=linear_model, top_k_score=top_k_score, X_test=X_test, y_test=y_test)
    
    def insert_parent(self, parent):
        self.parent = parent
    
    def __str__(self) -> str:
        """ Returns a string representation of the TreeNode. """
        indent = "  " * self.depth * 2
        info = f"{indent}Depth: {self.depth}\n" if self.cluster_node else ""

        if self.cluster_node:
            if len(self.children.values()) == 0:
                info += f"{indent}Cluster Info: {self.cluster_node}, No Child Clusters\n"
            else:
                n_cluster = ""
                for child_cluster in self.children.keys():
                    n_cluster += str(child_cluster) + " "
                    
                info += f"{indent}Cluster Info: {self.cluster_node}, Child Clusters: {n_cluster}\n"

        if self.linear_node:
            info += f"{indent}Classifier Info: {self.linear_node.__str__(indent)}\n"

        for child in self.children.values():
            info += str(child)  # Recursively print children

        return info
        
        
    