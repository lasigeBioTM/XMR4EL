import os
import pandas as pd

from collections import defaultdict

from sklearn.preprocessing import MultiLabelBinarizer


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

        return grouped_train_df["corpus_name"].tolist()  # Returns a list of concatenated strings
    
    @staticmethod
    def enconde_labels(labels_data):
        labels_list = [[label] for label in labels_data]
        mlb = MultiLabelBinarizer(sparse_output=True)
        labels_matrix = mlb.fit_transform(labels_list)
        return labels_matrix, mlb
    
    def load_data_labels_from_file(self, train_filepath, labels_filepath, truncate_data=0):
        # Load and group texts
        train_df = pd.read_csv(
            train_filepath, 
            header=None,
            names=["id", "text"],
            delimiter="\t",
        )
        
        train_df["text"] = train_df["text"].astype(str).fillna("")
        
        grouped_texts = train_df.groupby("id")["text"].apply(lambda x: " ".join(x)).reset_index()

        # Apply truncation if requested
        if truncate_data > 0:
            grouped_texts = grouped_texts.head(truncate_data)
        
        # Load labels and truncate to match ID count (either original or truncated)
        with open(labels_filepath, 'r') as f:
            labels = [line.strip() for line in f if line.strip()][:len(grouped_texts)]  # Truncate here
        
        # Proceed as before
        labels_list = [[label] for label in labels]
        
        mlb = MultiLabelBinarizer(sparse_output=True)
        labels_matrix = mlb.fit_transform(labels_list)
        
        return {
            'corpus': grouped_texts["text"].tolist(),
            'labels_matrix': labels_matrix,
            'label_encoder': mlb
        }

            
            
