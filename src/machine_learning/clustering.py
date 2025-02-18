import os
import pickle


class Clustering():

    def __init__(self, model=None, model_type=None, params=None):
        self.model = model
        self.model_type = model_type
        self.params = params

    def save(self, clustering_folder):
        """Save the model and its parameters to a file """
        os.makedirs(os.path.dirname(clustering_folder), exist_ok=True)
        with open(clustering_folder, 'wb') as fout:
            pickle.dump(self.__dict__, fout)

    @classmethod
    def load(cls, clustering_folder):
        """Load a saved model from a file."""
        if not os.path.exists(clustering_folder):
            raise FileNotFoundError(f"Clustering folder {clustering_folder} does not exist.")
        
        with open(clustering_folder, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model
    
    
    def model_type(self):
        return self.model_type
