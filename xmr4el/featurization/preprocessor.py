from collections import defaultdict
import os
import pandas as pd


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
    def load_data_labels_from_file(train_filepath, labels_filepath, truncate_data=0):
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

        # Check length match with new behavior
        if len(raw_labels) > len(grouped_texts):
            # Truncate labels to match texts length
            raw_labels = raw_labels[:len(grouped_texts)]
            print(f"Warning: Truncated labels from {len(raw_labels)} to {len(grouped_texts)} to match text groups")
        elif len(raw_labels) < len(grouped_texts):
            raise Exception(f"Mismatch: Not enough labels ({len(raw_labels)}) for text groups ({len(grouped_texts)})")

        # Step 4: Inject labels into grouped_texts
        grouped_texts["concept_id"] = raw_labels

        # Step 5: Keep original synonyms for cross-encoder
        grouped_texts["original_texts"] = grouped_texts["text"]

        return {
            "corpus": grouped_texts["original_texts"].tolist(),  # List[List[str]]
            "labels": grouped_texts["concept_id"].tolist()         # List[str]
        }

    @staticmethod
    def prepare_data(X_train, Y_train): # corpus, labels
        # Step 1: Flatten synonyms into corpus and build reverse mapping
        trn_corpus = []
        label_to_indices = defaultdict(list)  # label -> indices of its synonyms in trn_corpus

        for label, synonyms in zip(Y_train, X_train):
            for synonym in synonyms:
                idx = len(trn_corpus)
                trn_corpus.append(synonym)
                label_to_indices[label].append(idx)

        return trn_corpus, label_to_indices
    
    @staticmethod
    def prepare_label_matrix(label_to_indices):
        label_to_matrix = []
        
        for key in list(label_to_indices.keys()):
            labels_ids = label_to_indices[key]
            
            for _ in labels_ids:
                label_to_matrix.append([key])
        
        return label_to_matrix
            
