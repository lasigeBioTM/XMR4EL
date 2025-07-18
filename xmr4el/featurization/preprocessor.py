import os
import logging
import pandas as pd

from fuzzywuzzy import fuzz

from sklearn.preprocessing import MultiLabelBinarizer

# LOGGER = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )

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
        # Step 1: Load TSV (mentions file)
        train_df = pd.read_csv(
            train_filepath, 
            header=None,
            names=["group_id", "text"],
            delimiter="\t",
            dtype={"group_id": int, "text": str}
        )
        train_df["text"] = train_df["text"].fillna("")

        # Step 2: Group synonyms by group_id
        grouped_texts = train_df.groupby("group_id")["text"].apply(list).reset_index()

        # Step 3: Load labels.txt (concept codes)
        with open(labels_filepath, 'r') as f:
            raw_labels = [line.strip() for line in f if line.strip()]

        # Optional truncation (truncate both mentions and labels to same number)
        if truncate_data > 0:
            grouped_texts = grouped_texts.head(truncate_data)
            raw_labels = raw_labels[:truncate_data]

        # Check length match
        if len(raw_labels) != len(grouped_texts):
            raise Exception(f"Mismatch: {len(raw_labels)} labels vs {len(grouped_texts)} mention groups")

        # Step 4: Inject labels into grouped_texts
        grouped_texts["concept_id"] = raw_labels

        # Step 5: Keep original synonyms for cross-encoder
        grouped_texts["original_texts"] = grouped_texts["text"]

        return {
            "cross_corpus": grouped_texts["original_texts"].tolist(),  # List[List[str]]
            "raw_labels": grouped_texts["concept_id"].tolist()         # List[str]
        }

            
            
