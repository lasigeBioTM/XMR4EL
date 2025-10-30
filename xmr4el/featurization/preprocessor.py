import os
import logging

import pandas as pd

from typing import Dict, List, Sequence, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

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
            # logger.info(f"{label} -> {synonyms}")
            for synonym in synonyms:
                idx = len(trn_corpus)
                trn_corpus.append(synonym)
                label_to_indices[label].append(idx)

        return trn_corpus, label_to_indices
    
    @staticmethod
    def prepare_label_matrix(label_to_indices: Dict[str, List[int]]) -> List[List[str]]:
        """Expand a label-to-index mapping into a label matrix."""
        return [[key] for key, ids in label_to_indices.items() for _ in ids]
    
    @staticmethod
    def load_pubtator_for_mention_embeddings(pubtator_filepath: str) -> List[Dict[str, str]]:
        """
        Load a PubTator file and return a flat list of mentions ready for embedding-based EL.

        Each dict has:
            - 'mention_text': the mention string
            - 'context': full title + abstract
            - 'cui': gold concept ID
            - 'sem_types': list of semantic types
            - 'pmid': PMID
            - 'start', 'end': character offsets
        """
        assert os.path.exists(pubtator_filepath), f"{pubtator_filepath} does not exist"

        examples: List[Dict[str, str]] = []
        current_pmid = None
        title, abstract = "", ""

        with open(pubtator_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Title line
                if "|t|" in line:
                    parts = line.split("|", 2)
                    if len(parts) == 3:
                        current_pmid = parts[0]
                        title = parts[2]
                        abstract = ""

                # Abstract line
                elif "|a|" in line:
                    parts = line.split("|", 2)
                    if len(parts) == 3:
                        current_pmid = parts[0]
                        abstract = parts[2]

                # Annotation line
                elif "\t" in line:
                    parts = line.split("\t")
                    if len(parts) >= 6:
                        pmid, start, end, mention_text, sem_types, cui = parts[:6]
                        context = " ".join(x for x in [title, abstract] if x)
                        examples.append({
                            "mention_text": mention_text,
                            "context": context,
                            "cui": cui,
                            "sem_types": sem_types.split(",") if sem_types else [],
                            "pmid": pmid,
                            "start": int(start),
                            "end": int(end)
                        })

        return examples

    @staticmethod
    def load_pubtator_file(pubtator_filepath: str) -> Dict[str, List]:
        """
        Load a PubTator file and return corpus and labels lists for EL training.

        corpus[i] = list of "mention [SEP] context" for all mentions/synonyms of that entry
        labels[i] = CUI for that entry
        """
        assert os.path.exists(pubtator_filepath), f"{pubtator_filepath} does not exist"

        corpus: List[List[str]] = []
        labels: List[str] = []

        current_pmid = None
        title, abstract = "", ""
        mentions_for_current_pmid: List[str] = []
        cui_for_current_pmid = None

        with open(pubtator_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Title line
                if "|t|" in line:
                    parts = line.split("|", 2)
                    if len(parts) == 3:
                        # Save previous pmid info
                        if current_pmid is not None and mentions_for_current_pmid:
                            corpus.append(mentions_for_current_pmid)
                            labels.append(cui_for_current_pmid)

                        current_pmid = parts[0]
                        title = parts[2]
                        abstract = ""
                        mentions_for_current_pmid = []
                        cui_for_current_pmid = None

                # Abstract line
                elif "|a|" in line:
                    parts = line.split("|", 2)
                    if len(parts) == 3:
                        current_pmid = parts[0]
                        abstract = parts[2]

                # Annotation line
                elif "\t" in line:
                    parts = line.split("\t")
                    if len(parts) >= 6:
                        mention_text = parts[3]
                        context = " ".join(x for x in [title, abstract] if x)
                        cui = parts[5]

                        mentions_for_current_pmid.append(f"{mention_text} [SEP] {context}")
                        # Assign CUI for this pmid; assumes same CUI for all mentions (if multiple CUIs, you may need a strategy)
                        if cui_for_current_pmid is None:
                            cui_for_current_pmid = cui

            # Don't forget the last pmid
            if current_pmid is not None and mentions_for_current_pmid:
                corpus.append(mentions_for_current_pmid)
                labels.append(cui_for_current_pmid)

        return {"corpus": corpus, "labels": labels}