from src.models.cluster_wrapper.clustering_model import ClusteringModel


class ClusterWrapper():
    
    def __init__(self, cluster_model=None):
        """Initialization.
        
        Args:
            cluster_model (ClusterML): Clustering Algorithm
        """
        self.cluster_model = cluster_model
    
    def save(self, clustering_folder):
        """Save the clustering object to a folder

        Args:
            clustering_folder (str): The saving folder name
        """
        self.cluster_model.save(clustering_folder)
        
    @classmethod
    def load(cls, clustering_folder):
        """Load preprocessor

        Args:
            clustering_folder (str): The folder to load

        Returns:
            cls: An instance of ClusterWrapper
        """
        cluster_model = ClusteringModel.load(clustering_folder)
        return cls(cluster_model)
    
    
        
    