import numpy as np

from joblib import Parallel, delayed

from sklearn.metrics import silhouette_score, davies_bouldin_score

from xmr4el.models.cluster_wrapper.clustering_model import ClusteringModel


class XMRTuner:

    @staticmethod
    def tune_k(
        trn_corpus,
        config,
        dtype,
        k_range,
        weight_silhouette=0.5,
        weight_db=0.3,
        weight_elbow=0.2,
    ):
        """Optimize adaptive K selection with parallel processing."""

        def evaluate_k(k):
            """
                Trains a clustering model and computes
                evaluation metrics for a given k.
            """

            print(f"In evaluate_k with k={k}")

            model = ClusteringModel.train(
                trn_corpus,
                {**config, "kwargs": {**config["kwargs"], "n_clusters": k}},
                dtype=dtype,
            ).model.model
            labels = model.labels_

            num_clusters = len(set(labels))  # Count unique clusters

            if num_clusters <= 1:
                print(
                    f"Warning: Only {num_clusters} clusters found for k={k}, skipping..."
                )
                return k, np.inf, 0, np.inf  # Return bad scores

            inertia = model.inertia_
            sil_score = (
                silhouette_score(trn_corpus, labels, 
                                 metric="cosine", random_state=0)
                if num_clusters > 1
                else 0
            )
            db_score = (
                davies_bouldin_score(trn_corpus, labels) if num_clusters > 1 else np.inf
            )

            return k, inertia, sil_score, db_score

        corpus_len = len(trn_corpus)

        if k_range[1] > corpus_len:
            k_range = (k_range[0], corpus_len)
            
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_k)(k) for k in range(k_range[0], k_range[1] + 1)
        )

        # Extract valid results (filter out np.inf)
        results = [res for res in results if res[1] != np.inf]
        if not results:  # If all k values failed, return default
            print("No valid clustering found, returning k=2")
            return 2, np.zeros(len(k_range))  # Default k

        ks, inertia_scores, silhouette_scores, davies_bouldin_scores = zip(*results)
        inertia_scores, silhouette_scores, davies_bouldin_scores = map(
            np.array, [inertia_scores, silhouette_scores, davies_bouldin_scores]
        )

        # Normalize scores
        inertia_scores = 1 - (inertia_scores - inertia_scores.min()) / (
            inertia_scores.max() - inertia_scores.min()
        )
        silhouette_scores = (silhouette_scores - silhouette_scores.min()) / (
            silhouette_scores.max() - silhouette_scores.min()
        )
        davies_bouldin_scores = 1 - (
            davies_bouldin_scores - davies_bouldin_scores.min()
        ) / (davies_bouldin_scores.max() - davies_bouldin_scores.min())

        # Compute weighted score
        combined_score = (
            (weight_elbow * inertia_scores)
            + (weight_silhouette * silhouette_scores)
            + (weight_db * davies_bouldin_scores)
        )

        # Select the best k
        best_k = ks[np.argmax(combined_score)]

        return int(best_k), combined_score
