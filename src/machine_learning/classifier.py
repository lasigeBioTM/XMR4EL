import os
import pickle

class Classifier():

    def __init__(self, model=None, model_type=None, params=None):
        self.model = model
        self.model_type = model_type
        self.params = params

    def save(self, classifier_folder):
        """Save the model and its parameters to a file """
        os.makedirs(os.path.dirname(classifier_folder), exist_ok=True)
        with open(classifier_folder, 'wb') as fout:
            pickle.dump(self.__dict__, fout)

    @classmethod
    def load(cls, classifier_folder):
        """Load a saved model from a file."""
        if not os.path.exists(classifier_folder):
            raise FileNotFoundError(f"Classifier folder {classifier_folder} does not exist.")
        
        with open(classifier_folder, 'rb') as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model