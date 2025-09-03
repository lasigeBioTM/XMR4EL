import os
import pandas as pd

from typing import Dict, List, Sequence, Tuple

from collections import defaultdict

class Preprocessor:
    """Preprocess text to numerical values."""

    @staticmethod
    def load_data_from_file(train_filepath: str) -> List[str]:
        """Load the training data from a TSV file."""
        assert os.path.exists(train_filepath), f"{train_filepath} does not exist"

        train_df = pd.read_csv(
            train_filepath,
            header=None,
            names=["id", "corpus_name"],
            delimiter="\t",
        )

        train_df["corpus_name"] = train_df["corpus_name"].astype(str).fillna("")

        grouped_train_df = (
            train_df.groupby("id")["corpus_name"].apply(lambda x: " ".join(x)).reset_index()
        )

        return grouped_train_df["corpus_name"].tolist()
    
    @staticmethod
    def load_data_labels_from_file(
        train_filepath: str,
        labels_filepath: str,
        truncate_data: int = 0,
    ) -> Dict[str, List[List[str]]]:
        """Load texts and their corresponding labels from file."""
        
        train_df = pd.read_csv(
            train_filepath,
            header=None,
            names=["group_id", "text"],
            delimiter="\t",
            dtype={"group_id": int, "text": str},
        )
        
        train_df["text"] = train_df["text"].fillna("")

        grouped_texts = train_df.groupby("group_id")["text"].apply(list).reset_index()
        
        with open(labels_filepath, 'r') as f:
            raw_labels = [line.strip() for line in f if line.strip()]

        if truncate_data > 0:
            grouped_texts = grouped_texts.head(truncate_data)
            raw_labels = raw_labels[:truncate_data]

        if len(raw_labels) > len(grouped_texts):
            raw_labels = raw_labels[:len(grouped_texts)]
            print(
                f"Warning: Truncated labels from {len(raw_labels)} to {len(grouped_texts)} to match text groups"
            )
        elif len(raw_labels) < len(grouped_texts):
            raise Exception(
                f"Mismatch: Not enough labels ({len(raw_labels)}) for text groups ({len(grouped_texts)})"
            )

        grouped_texts["concept_id"] = raw_labels

        grouped_texts["original_texts"] = grouped_texts["text"]

        return {
            "corpus": grouped_texts["original_texts"].tolist(),
            "labels": grouped_texts["concept_id"].tolist(),
        }

    @staticmethod
    def prepare_data(
        X_train: Sequence[Sequence[str]],
        Y_train: Sequence[str],
    ) -> Tuple[List[str], Dict[str, List[int]]]:
        """Flatten synonyms and build reverse label index mapping."""
        trn_corpus: List[str] = []
        label_to_indices: Dict[str, List[int]] = defaultdict(list)

        for label, synonyms in zip(Y_train, X_train):
            for synonym in synonyms:
                idx = len(trn_corpus)
                trn_corpus.append(synonym)
                label_to_indices[label].append(idx)

        return trn_corpus, label_to_indices
    
    @staticmethod
    def prepare_label_matrix(label_to_indices: Dict[str, List[int]]) -> List[List[str]]:
        """Expand a label-to-index mapping into a label matrix."""
        label_to_matrix: List[List[str]] = []
        
        for key in list(label_to_indices.keys()):
            labels_ids = label_to_indices[key]
            
            for _ in labels_ids:
                label_to_matrix.append([key])
        
        return label_to_matrix
            
