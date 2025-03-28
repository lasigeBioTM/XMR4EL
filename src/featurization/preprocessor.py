import os
import pandas as pd


class Preprocessor:
    """Preprocess text to numerical values"""

    @staticmethod
    def load_data_from_file(train_filepath):
        """Load the training data and labels data

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
