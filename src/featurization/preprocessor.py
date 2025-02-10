import json
import pickle
import os
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, load_npz

kb_dict = {
    'medic': 'DiseaseID',
    'chemical': 'ChemicalID'
    }

class Preprocessor():

    def __init__(self, model=None, model_type=None):
        self.model = model
        self.model_type = model_type

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        model_data = {
            'model': self.model, 
            'model_type': self.model_type
        }
        with open(os.path.join(directory, 'vectorizer.pkl'), 'wb') as fout:
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
        
    @staticmethod
    def save_biobert_labels(all_embeddings, directory, filepath):
        # assert os.path.exists(directory),f"{directory} does not exist" 
        np.savez_compressed(filepath, embeddings=all_embeddings)
    
    @staticmethod
    def load_biobert_labels(directory):
        assert os.path.exists(directory),f"{directory} does not exist"
        return np.load(directory)['embeddings']
    
    @staticmethod
    def load_labels_from_file(labels_folder):
        labels_file = os.path.join(labels_folder, 'labels.json')

        with open(labels_file, 'r') as json_file:
            labels_data = json.load(json_file)

        labels_dict = {}
        for labels_id, entries in labels_data.items():
            names_and_synonyms = " ".join(entries)
            unique_words = list(set(names_and_synonyms.split()))
            combined_text = ' '.join(unique_words)
            labels_dict[labels_id] = combined_text
        
        processed_labels_data = list(labels_dict.values())
        processed_labels_id = list(labels_dict.keys())

        return (processed_labels_id, processed_labels_data)
    
    @staticmethod
    def load_labels_from_dict(labels_data):
        labels_dict = {}
        for labels_id, entries in labels_data.items():
            names_and_synonyms = " ".join(entries)
            unique_words = list(set(names_and_synonyms.split()))
            combined_text = ' '.join(unique_words)
            labels_dict[labels_id] = combined_text
        
        processed_labels_data = list(labels_dict.values())
        processed_labels_id = list(labels_dict.keys())

        return (processed_labels_id, processed_labels_data)
    
    @staticmethod 
    def load_data_from_file(train_filepath, labels_filepath):
        assert os.path.exists(train_filepath), f"{train_filepath} does not exist"
        assert os.path.exists(labels_filepath), f"{labels_filepath} does not exist"

        train_df = pd.read_csv(train_filepath, header=None, names=['id', 'corpus_name'], delimiter="\t")

        grouped_train_df = train_df.groupby('id')['corpus_name'].apply(list).reset_index()

        labels_df = pd.read_csv(labels_filepath, header=None, names=['id'], delimiter="\t")

        return {
            "labels": labels_df['id'].tolist(),
            "corpus": grouped_train_df['corpus_name'].tolist()
        }
    
        