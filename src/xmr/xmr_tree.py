class XMRTree():
    
    def __init__(self,
                 text_embeddings,
                 transformer_embeddings,
                 concatenated_embeddings,
                 clustering_model,
                 classifier_model,
                 test_split,
                 children={}, # XMRTree
                 depth=0
                 ):
        
        self.text_embeddings = text_embeddings
        self.transformer_embeddings = transformer_embeddings
        self.concatenated_embeddings = concatenated_embeddings
        
        self.clustering_model = clustering_model
        
        self.classifier_model = classifier_model
        self.test_split = test_split
        
        self.children = children
        self.depth = depth
        
    