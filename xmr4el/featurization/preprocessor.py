import os
import pandas as pd
from scipy.sparse import csr_matrix

class Preprocessor:
    """Preprocess text to numerical values"""

    @staticmethod
    def load_data_from_file(train_filepath):
        """Load the training data

        Args:
            train_filepath (str): Path to the training data

        Returns:
            corpus (list): List containing all the training data as
            concatenated strings.
        """

        assert os.path.exists(train_filepath), f"{train_filepath} does not exist"

        train_df = pd.read_csv(
            train_filepath,
            header=None,
            names=["id", "corpus_name"],
            delimiter="\t"
        )

        train_df["corpus_name"] = train_df["corpus_name"].astype(str).fillna("")

        # Merge multiple corpus names per ID into a single string
        grouped_train_df = (
            train_df.groupby("id")["corpus_name"]
            .apply(lambda x: " ".join(x))
            .reset_index()
        )

        return grouped_train_df[
            "corpus_name"
        ].tolist()  # Returns a list of concatenated strings
        
    def load_data_labels_from_file(self, train_filepath, labels_filepath):
        """Load the training data and labels data

        Args:
            train_filepath (str): Path to the training data
            label_filepath (str): Path to the labels data

        Returns:
            corpus (dict): Dictionary with an sparse matrix of labels and corpus,
            the corpus is a list of concatenated strings
        """
        
        assert os.path.exists(labels_filepath), f"{labels_filepath} does not exist"
        
        labels_data = []
        
        with open(labels_filepath, 'r') as fin:
            for line in fin:
                labels_data.append(line)
        
        train_data = self.load_data_from_file(train_filepath)
        
        return {'corpus': train_data, 'labels_matrix': csr_matrix(labels_data)}
        
        
