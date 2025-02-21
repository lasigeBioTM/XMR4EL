class ClusterNode:
    def __init__(self, model=None, labels=None, cluster_points=None):
        self.model = model
        self.labels = labels
        self.cluster_points = cluster_points

    def print_out(self, output=""):
        """
        Returns a string representation of the Node object.
        Includes information about the clustering model and labels.
        """
        n_clusters = len(set(self.labels)) if self.labels is not None else 0
        return f"Model: {type(self.model).__name__}, Clusters: {n_clusters}"

class LinearNode:
    
    def __init__(self, model=None, top_k_score=None, X_test=None, y_test=None):
        self.model = model
        self.top_k_score = top_k_score
        self.X_test = X_test
        self.y_test = y_test
        
    def print_out(self, output=""):
        out = f"{output}Linear Model: {type(self.model).__name__}\n"
        out += f"{output}Top-K Score: {self.top_k_score}\n"
        out += f"{output}X_test Shape: {self.X_test.shape}\n"
        return out
        
    
class TreeNode:
    def __init__(self, depth=0, cluster_node=None, linear_node=None, child=None):
        self.depth = depth
        self.cluster_node = cluster_node
        self.child = child if child is not None else []
        self.overall_silhouette_score = None
        self.silhouette_scores = {}
        self.linear_node = linear_node
    
    def insert_child(self, child):
        self.child.append(child)
        
    def insert_cluster_node(self, clustering_model, cluster_labels, cluster_points):
        self.cluster_node = ClusterNode(model=clustering_model, labels=cluster_labels, cluster_points=cluster_points)
        
    def insert_linear_node(self, linear_model, top_k_score, X_test, y_test):
        self.linear_node = LinearNode(model=linear_model, top_k_score=top_k_score, X_test=X_test, y_test=y_test)
    
    def insert_overall_silhouette_scores(self, overall_silhouette_score):
        self.overall_silhouette_score = overall_silhouette_score
        
    def insert_silhouette_scores_dict(self, silhouette_scores):
        self.silhouette_scores = silhouette_scores
    
    def print_tree(self, level=0, linear=True, cluster=True):
        """
        Recursively prints the tree structure.
        Each node calls its `print_out` method for its representation.
        """
        output = "  " * level  # Indentation for current level
        out = "" if self.cluster_node is None else f"{output}Depth: {self.depth}\n"
        
        if self.cluster_node is not None and cluster:
            silhouette_score_out = "\n"
                
            for idx, score in self.silhouette_scores.items():
                silhouette_score_out += f"{output}{int(idx)}: {float(score)}\n"
                
            out += f"{output}Model: {self.cluster_node.print_out()}, \n"
            out += f"{output}Overall Silhouette Score: {self.overall_silhouette_score}\n"
            out += f"{output}Samples Silhouette Scores: {silhouette_score_out}\n"
            
        if self.linear_node is not None and linear:
            out += f"{self.linear_node.print_out(output=output)}\n"            

        # Recursively print child nodes
        for child in self.child:
            out += child.print_tree(level + 5, linear=linear, cluster=cluster)

        return out


            
        
    
        
        
        
    