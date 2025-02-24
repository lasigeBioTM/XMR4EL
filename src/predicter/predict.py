import numpy as np

class PredictTopK():
        
    @classmethod
    def predict(cls, tree_node, X_input, k=3, max_depth=None):
        """
        Recursively traverses the hierarchical tree to classify an input embedding.
        If a leaf node has no classifier, it moves up the tree to find the nearest classifier.

        Parameters:
        - tree_node: The current node in the hierarchical tree.
        - X_input: The input embedding to classify.
        - k: Number of top predictions to return.
        - max_depth: The depth where classifiers exist.

        Returns:
        - A tuple (top_k_labels, top_k_confidences, cluster_label)
        """
        
        # 🛑 Check if tree_node is missing a cluster model
        if tree_node.cluster_node is None:
            # print(f"[WARNING] Node at depth {tree_node.depth} has no cluster model, using classifier.")
            return cls.classify_at_nearest_valid_node(tree_node, X_input, k)

        # Step 1: Predict cluster label
        cluster_model = tree_node.cluster_node.model
        cluster_label = cluster_model.predict(X_input)[0]

        # Debugging prints
        # print(f"Depth: {tree_node.depth}, Number of Children: {len(tree_node.child)}, Cluster Label: {cluster_label}")

        # Step 2: If a child node exists, continue traversal
        child_node = tree_node.children.get(cluster_label)
        if child_node:
            return cls.predict(child_node, X_input, k, max_depth)

        # Step 3: If no children, determine if this node should have a classifier
        if max_depth is not None and tree_node.depth < max_depth:
            # print(f"[WARNING] No classifier at depth {tree_node.depth}, moving up the tree to find one...")
            return cls.classify_at_nearest_valid_node(tree_node, X_input, k)

        # Step 4: Use the classifier from the current node
        return cls.classify_at_nearest_valid_node(tree_node, X_input, k)


    @classmethod
    def classify_at_nearest_valid_node(cls, tree_node, X_input, k):
        """
        Finds the nearest valid node with a classifier and classifies X_input.
        """
        while tree_node is not None:
            if tree_node.linear_node is not None:
                classifier = tree_node.linear_node.model
                probs = classifier.predict_proba(X_input)[0]  # Get class probabilities

                # Get top-k indices sorted by confidence
                top_k_indices = np.argsort(probs)[::-1][:k]
                top_k_labels = classifier.classes_[top_k_indices].tolist()  # Retrieve corresponding labels
                top_k_confidences = probs[top_k_indices].tolist()  # Retrieve corresponding confidence scores

                return top_k_labels, top_k_confidences, tree_node.depth

            # Move up the tree (you need a way to track the parent)
            tree_node = tree_node.parent

        # 🚨 If no classifier was found, raise an error
        raise RuntimeError("No valid classifier found in the entire tree!")


    # Batch Prediction
    @classmethod
    def predict_batch(cls, tree_node, X_input, k=3, batch_size=256, max_depth=None):
        """
        Predicts top-k labels, confidence scores, and true labels for a batch of input embeddings.

        Parameters:
        - tree_node: The root TreeNode containing the hierarchical model.
        - X_input: A dense array (NumPy array) of input embeddings, shape (n_samples, embedding_dim).
        - k: The number of top predictions to return.
        - batch_size: Number of samples to process per batch.

        Returns:
        - predictions: A list of dictionaries, each containing:
            - 'top_k_labels': List of top-k predicted labels.
            - 'top_k_confidence': Corresponding confidence scores.
            - 'true_label': The ground truth cluster label from the leaf node.
        """
        predictions = []
        n_samples = X_input.shape[0]

        # print(X_input.shape)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_embeddings = X_input[start:end]  # Extract batch
            # print(f"Processing batch: {start}-{end}")

            for input_embedding in batch_embeddings:
                input_embedding = input_embedding.reshape(1, -1)
                # print("Started Predict")
                top_k_labels, top_k_confidence, true_label = cls.predict(tree_node, input_embedding, k=k, max_depth=max_depth)
                predictions.append({
                    'top_k_labels': top_k_labels,
                    'top_k_confidence': top_k_confidence,
                    'true_label': true_label
                })
                # print("Ended predict")

        return predictions

    @staticmethod
    def __temperature_scale(probs, T=2.0):
        """Applies temperature scaling to soften confidence scores."""
        probs = np.array(probs)  # Ensure it's a NumPy array
        scaled_logits = np.log(probs + 1e-9) / T  # Apply temperature scaling
        exp_logits = np.exp(scaled_logits)
        return exp_logits / np.sum(exp_logits)  # Normalize

    @staticmethod
    def __label_smoothing(probs, alpha=0.1):
        """Applies label smoothing to prevent overconfidence."""
        num_classes = len(probs)
        return (1 - alpha) * np.array(probs) + alpha / num_classes  # Adjust confidence scores
    
    @staticmethod
    def compute_ece(predicted_batch, num_bins=10):
        """
        Compute Expected Calibration Error (ECE) for confidence scores.
        
        Parameters:
        - predicted_batch: List of dictionaries with 'top_k_confidences' and 'true_label'.
        - num_bins: Number of bins to group confidence scores.
        
        Returns:
        - ece: The Expected Calibration Error (lower is better)
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)  # Bins from 0 to 1
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        confidences = []
        accuracies = []

        for prediction in predicted_batch:
            if prediction["true_label"] is not None and prediction["top_k_confidence"]:
                max_confidence = max(prediction["top_k_confidence"])  # Top-1 confidence
                is_correct = int(prediction["true_label"] in prediction["top_k_labels"])  # 1 if correct, 0 otherwise

                confidences.append(max_confidence)
                accuracies.append(is_correct)

        confidences = np.array(confidences)
        accuracies = np.array(accuracies)

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(accuracies[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                ece += np.abs(bin_confidence - bin_accuracy) * (np.sum(in_bin) / len(confidences))

        return ece