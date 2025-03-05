import os
import pandas as pd

from src.featurization.bert_vectorizer import BertVectorizer


class BertPreprocessor():

    def __init__(self, bert_vectorizer=None):
        """Initialization

        Args:
            bert_vectorizer (BertVectorizer): Text BERT vectorizer class instance.
        """
        
        self.bert_vectorizer = bert_vectorizer

    def save(self, bert_vectorizer_folder):
        """Save the preprocess object to a folder

        Args:
            bert_preprocessor_folder (str): The saving folder name
        """
        self.bert_vectorizer.save(bert_vectorizer_folder)

    @classmethod
    def load(cls, bert_preprocessor_folder):
        """Load preprocessor

        Args:
            bert_preprocess_folder (str): The folder to load

        Returns:
            cls: An instance of BertPreprocessor
        """
        
        bert_vectorizer = BertVectorizer.load(bert_preprocessor_folder)
        return cls(bert_vectorizer)
    
    @staticmethod 
    def load_data_from_file(train_filepath, labels_filepath):
        """Load the training data and labels data

        Args:
            train_filepath (str): Path to the training data
            labels_filepath (str): Path to the labels data

        Returns:
            data_dict (dict): 
                labels (lst): List containing all the labels
                corpus (lst): List containing all the training data
        """
        
        assert os.path.exists(train_filepath), f"{train_filepath} does not exist"
        assert os.path.exists(labels_filepath), f"{labels_filepath} does not exist"

        train_df = pd.read_csv(train_filepath, header=None, names=['id', 'corpus_name'], delimiter="\t")

        grouped_train_df = train_df.groupby('id')['corpus_name'].apply(list).reset_index()

        labels_df = pd.read_csv(labels_filepath, header=None, names=['id'], delimiter="\t")

        data_dict = {
            "labels": labels_df['id'].tolist(),
            "corpus": grouped_train_df['corpus_name'].tolist()
        }
        
        return data_dict
    
        