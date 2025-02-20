class Node:
    def __init__(self, model=None, labels=None):
        self.model = model
        self.labels = labels

    def print_out(self):
        """
        Returns a string representation of the Node object.
        Includes information about the clustering model and labels.
        """
        n_clusters = len(set(self.labels)) if self.labels is not None else 0
        return f"Model: {type(self.model).__name__}, Clusters: {n_clusters}"

    
class TreeNode:
    def __init__(self, depth=0, node=None, child=None):
        self.depth = depth
        self.node = node
        self.child = child if child is not None else []
        self.overall_silhouette_score = None
        self.silhouette_scores = {}
        self.linear_model = None
        
    def insert_child(self, child):
        self.child.append(child)
        
    def insert_node(self, node):
        self.node = node
        
    def insert_linear_model(self, linear_model):
        self.linear_model = linear_model
    
    def insert_overall_silhouette_scores(self, overall_silhouette_score):
        self.overall_silhouette_score = overall_silhouette_score
        
    def insert_silhouette_scores_dict(self, silhouette_scores):
        self.silhouette_scores = silhouette_scores

    def print_tree(self, level=0):
        """
        Recursively prints the tree structure.
        Each node calls its `print_out` method for its representation.
        """
        output = "  " * level
        if self.node is not None:
            
            silhouette_score_out = "\n"
            
            for idx, score in self.silhouette_scores.items():
                silhouette_score_out += f"{3*output}{int(idx)}: {float(score)}\n"
            
            output += f"Depth: {self.depth}, Model: {self.node.print_out()}, \n{output}Overall Silhouette Score: {self.overall_silhouette_score}\n{output}Samples Silhouette Scores: {silhouette_score_out}\n"
        else:
            return ""  # Skip printing if there's no data in the node

        # Recursively print child nodes
        for child in self.child:
            output += child.print_tree(level + 5)

        return output

            
        
    
        
        
        
    