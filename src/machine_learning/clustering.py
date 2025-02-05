import os
import pickle


class Clustering():

    def __init__(self, model=None, model_type=None):
        self.model = model
        self.model_type = model_type

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        model_data = {
            'model': self.model, 
            'model_type': self.model_type
        }
        with open(os.path.join(directory, 'clustering.pkl'), 'wb') as fout:
            pickle.dump(model_data, fout)

    @classmethod
    def load(cls, model_path):
        assert os.path.exists(model_path), f"{model_path} does not exist"
        with open(model_path, 'rb') as fclu:
            data = pickle.load(fclu)
        return cls(
            model=data['model'], 
            model_type=data['model_type']
        )
    
    
    def model_type(self):
        return self.model_type
