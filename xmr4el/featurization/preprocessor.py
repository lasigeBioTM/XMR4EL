import os
import logging
import pandas as pd

from fuzzywuzzy import fuzz

from sklearn.preprocessing import MultiLabelBinarizer

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

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
        # Load and group texts (original code)
        train_df = pd.read_csv(
            train_filepath, 
            header=None,
            names=["id", "text"],
            delimiter="\t",
        )
        train_df["text"] = train_df["text"].astype(str).fillna("")
        grouped_texts = train_df.groupby("id")["text"].apply(list).reset_index()

        # --- NEW: Deduplicate texts for `corpus` only (keep `cross_corpus` untouched) ---
        def deduplicate_texts(texts, similarity_threshold=85):
            """Keep only semantically distinct texts per ID using fuzzy matching."""
            unique_texts = []
            for text in sorted(texts, key=len, reverse=True):  # Longest first
                if not any(fuzz.ratio(text, existing) >= similarity_threshold 
                        for existing in unique_texts):
                    unique_texts.append(text)
            return " ".join(unique_texts)  # Return merged string for corpus

        # Apply deduplication to `joined_text` (for TF-IDF/clustering)
        # grouped_texts["joined_text"] = grouped_texts["text"].apply(deduplicate_texts)
        
        # Keep original texts for `cross_corpus` (untouched for cross-encoder)
        grouped_texts["original_texts"] = grouped_texts["text"]  # Backup for cross_corpus

        # Truncate data if requested (original code)
        if truncate_data > 0:
            grouped_texts = grouped_texts.head(truncate_data)

        # Load labels (original code)
        with open(labels_filepath, 'r') as f:
            if truncate_data > 0:
                labels = [line.strip() for line in f if line.strip()][:truncate_data]
            else:
                labels = [line.strip() for line in f if line.strip()]
            
            if len(labels) > len(grouped_texts):
                LOGGER.warning("Labels and Train length mismatch, Labels > Train. Truncating Labels. "
                            f"Labels length: {len(labels)}, Train length: {len(grouped_texts)}")
                labels = labels[:len(grouped_texts)]
            elif len(labels) < len(grouped_texts):
                raise Exception(f"Labels and Train length mismatch, Train > Labels. Exiting. Labels: {len(labels)}, Train: {len(grouped_texts)}")

        # Prepare labels (original code)
        # labels_list = [[label] for label in labels]
        # mlb = MultiLabelBinarizer(sparse_output=True)
        # labels_matrix = mlb.fit_transform(labels_list)

        return {
            # 'corpus': grouped_texts["joined_text"].tolist(),      # Deduplicated for TF-IDF
            'cross_corpus': grouped_texts["original_texts"].tolist(),  # Original for cross-encoder
            'raw_labels': labels,
            # 'labels# _matrix': labels_matrix,
            # 'label_encoder': mlb
        }

            
            
