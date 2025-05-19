from xmr4el.xmr.tree import XMRTree

from sklearn.metrics import top_k_accuracy_score


def main():
    # train_disease_100 + test_data == true_labels
    test_xtree = XMRTree.load("test/test_data/saved_trees/XMRTree_2025-04-01_15-02-26")
    
    k = 3
    
    def test_top_k(xtree, k=5):
        # If no test split, return None
        if xtree.test_split is None:
            print(f"Node at depth {xtree.depth} has no test split.")
            return None        
        
        # Unpack the test split (X_test, y_test)
        X_test, y_test = xtree.test_split['X_test'], xtree.test_split['y_test']
        
        # Ensure the classifier model is available
        if xtree.classifier_model is None:
            print(f"Node at depth {xtree.depth} has no classifier model.")
            return None
        
        # Make predictions with the classifier model
        y_pred_proba = xtree.classifier_model.predict_proba(X_test)
        
        # Compute top-k accuracy score for the current node
        top_k_acc = top_k_accuracy_score(y_test, y_pred_proba, k=k)
        print(f"Top-{k} accuracy at depth {xtree.depth}: {top_k_acc}")
        
        # If there are children, compute top-k accuracy for each child recursively
        if xtree.children:
            for child_key, child in xtree.children.items():
                child_top_k_acc = test_top_k(child, k=k)
                print(f"Top-{k} accuracy for child {child_key} at depth {child.depth}: {child_top_k_acc}")
        
        return top_k_acc
    
    print("\n\n\n\n\n",test_xtree)
    
    test_top_k(test_xtree, k=k)

if __name__ == "__main__":
    main()