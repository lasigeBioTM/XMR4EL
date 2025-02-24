import numpy as np

class PredictTopK():
    
    # Inference
    @classmethod
    def predict(cls, tree_node, X_input, k=3):
        """
        Recursively traverses the hierarchical tree to classify an input embedding.

        Parameters:
        - X_input: The input embedding to classify.
        - k: Number of top predictions to return.

        Returns:
        - A tuple (top_k_labels, top_k_confidences)
        """
        if tree_node is None or tree_node.cluster_node is None:
            return [], [], None # No predictions possible
        
        cluster_model = tree_node.cluster_node.model
        cluster_label = cluster_model.predict(X_input)[0]
            
        # Find the matching child node
        for child in tree_node.child:
            if child.cluster_node and cluster_label in child.cluster_node.labels:
                return cls.predict(child, X_input, k) # Recursively call on the child node
        
        # Step 2: If no further children, use the local classifier for final classification
        if tree_node.linear_node is not None:
            classifier = tree_node.linear_node.model
            probs = classifier.predict_proba(X_input)[0] # Get class probabilities
            
            # Get top-k indices sorted by confidence
            top_k_indices = np.argsort(probs)[::-1][:k]
            top_k_labels = classifier.classes_[top_k_indices]  # Retrieve corresponding labels
            top_k_confidences = probs[top_k_indices]  # Retrieve corresponding confidence scores
            
            return top_k_labels.tolist(), top_k_confidences.tolist(), cluster_label
        
        # If no classifier is found, return None
        return [], [], cluster_label
    
    
    # True label should be the final label at the leaf value
    @classmethod
    def predict_batch(cls, tree_node, X_input, k=3, batch_size=256):
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
        - 'true_label': The ground truth cluster label from the leaf node.
        """
        predictions = []
        n_samples = X_input.shape[0]

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_embeddings = X_input[start:end]  # Extract batch

            if tree_node is None or tree_node.cluster_node is None:
                predictions.extend([{"top_k_labels": [], "top_k_confidences": [], "true_label": None}] * len(batch_embeddings))
                continue

            # Get cluster assignments for the batch
            cluster_labels = tree_node.cluster_node.model.predict(batch_embeddings)

            for i, input_embedding in enumerate(batch_embeddings):
                cluster_label = cluster_labels[i]
                
                # Find the correct subtree based on clustering
                child_node = None
                for child in tree_node.child:
                    if child.cluster_node and cluster_label in child.cluster_node.labels:
                        child_node = child
                        break

                # Recursively process if there's a matching child
                if child_node:
                    result = cls.predict_batch(child_node, np.array([input_embedding]), k)[0]
                    predictions.append(result)
                    continue

                # If no more children, classify using logistic regression
                if tree_node.linear_node is not None:
                    classifier = tree_node.linear_node.model
                    probs = classifier.predict_proba(input_embedding.reshape(1, -1))[0]  # Get probability scores

                    # Apply Temperature Scaling
                    # probs = cls.__temperature_scale(probs, T=2)
                    
                    # Apply Label Smoothing
                    # probs = cls.__label_smoothing(probs, alpha=0.1)                    
                    
                    top_k_indices = np.argsort(probs)[-k:][::-1]  # Get indices of top-k labels
                    top_k_labels = classifier.classes_[top_k_indices]  # Get top-k predicted labels
                    top_k_confidences = probs[top_k_indices]  # Get top-k confidence scores

                    predictions.append({
                        "top_k_labels": top_k_labels.tolist(),
                        "top_k_confidences": top_k_confidences.tolist(),
                        "true_label": cluster_label
                    })
                else:
                    predictions.append({"top_k_labels": [], "top_k_confidences": [], "true_label": cluster_label})
                    
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
            if prediction["true_label"] is not None and prediction["top_k_confidences"]:
                max_confidence = max(prediction["top_k_confidences"])  # Top-1 confidence
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