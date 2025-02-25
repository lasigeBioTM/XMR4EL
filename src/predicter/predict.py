import numpy as np

from collections import Counter


class PredictTopK():
        
    @classmethod
    def predict(cls, cluster_predictions, root_node, X_input, k=3):
        """
        Recursively traverses the hierarchical tree to classify an input embedding.
        """
        predictions = []
        for cluster_pred in cluster_predictions:
            current_node = root_node
            
            for idx in range(len(cluster_pred) + 1):
                if not current_node.is_leaf():
                    current_node = current_node.children[cluster_pred[idx]]
                else:
                    top_k_label, top_k_confidences = cls.classify(current_node.linear_node.model, X_input, k=k)
                    predictions.append({
                        'predicted_labels': top_k_label,
                        'top_k_confidences': top_k_confidences,
                        'true_label': cluster_pred[-1]
                    })
                
        return predictions
                
    @classmethod
    def classify(cls, linear_model, X_input, k=3):
        
        probs = linear_model.predict_proba(X_input)[0]  # Get class probabilities

        # Get top-k indices sorted by confidence
        top_k_indices = np.argsort(probs)[::-1][:k]
        top_k_labels = linear_model.classes_[top_k_indices].tolist()  # Retrieve corresponding labels
        top_k_confidences = probs[top_k_indices].tolist()  # Retrieve corresponding confidence scores

        return top_k_labels, top_k_confidences

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