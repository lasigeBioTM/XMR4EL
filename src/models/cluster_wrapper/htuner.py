import numpy as np

from joblib import Parallel, delayed

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

from src.models.cluster_wrapper.clustering_model import ClusteringModel


class KTuner():
    
    @staticmethod
    def tune(trn_corpus, config, dtype, k_range, weight_silhouette=0.5, weight_db=0.3, weight_elbow=0.2):
        """ Optimize adaptive K selection with parallel processing."""
        
        def evaluate_k(k):
            """Trains a clustering model and computes evaluation metrics for a given k."""
            
            print(f"In evaluate_k with k={k}")
            
            model = ClusteringModel.train(
                trn_corpus, 
                {
                    **config.model,  
                    'kwargs': {**config.model['kwargs'], 'n_clusters': k}  
                }, 
                dtype=dtype
            ).model.model
            labels = model.labels_            
            
            inertia = model.inertia_
            sil_score = silhouette_score(trn_corpus, labels, metric='cosine', random_state=0) if k > 1 else 0
            db_score = davies_bouldin_score(trn_corpus, labels) if k > 1 else np.inf
        
            return k, inertia, sil_score, db_score
        
        results = Parallel(n_jobs=-1)(delayed(evaluate_k)(k) for k in range(k_range[0], k_range[1] + 1))
            
        # Extract results
        ks, inertia_scores, silhouette_scores, davies_bouldin_scores = zip(*results)
        inertia_scores, silhouette_scores, davies_bouldin_scores = map(np.array, 
            [inertia_scores, silhouette_scores, davies_bouldin_scores])

        # Normalize scores
        inertia_scores = 1 - (inertia_scores - inertia_scores.min()) / (inertia_scores.max() - inertia_scores.min())
        silhouette_scores = (silhouette_scores - silhouette_scores.min()) / (silhouette_scores.max() - silhouette_scores.min())
        davies_bouldin_scores = 1 - (davies_bouldin_scores - davies_bouldin_scores.min()) / (davies_bouldin_scores.max() - davies_bouldin_scores.min())

        # Compute weighted score
        combined_score = (weight_elbow * inertia_scores) + (weight_silhouette * silhouette_scores) + (weight_db * davies_bouldin_scores)

        # Select the best k
        best_k = ks[np.argmax(combined_score)]

        return int(best_k), combined_score
    
    
class KeywordBasedAssessment():
    
    @staticmethod
    def get_representive_docs(trn_corpus, cluster_labels, top_n=3):
        """Find the most representive document in each cluster."""    
        unique_clusters = np.unique(cluster_labels)
        cluster_representatives = {}
        
        for cluster in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_embeddings = trn_corpus[cluster_indices]
            
            # Compute centroid 
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Compute cosine similarity between centroid and cluster members
            similarities = cosine_similarity([centroid], cluster_embeddings)[0]
            
            # Get top_n most representative documents
            top_indices = cluster_indices[np.argsort(similarities)[-top_n:]]
            cluster_representatives[cluster] = top_indices
    
        return cluster_representatives
    
    @staticmethod
    def get_cluster_keywords(texts, labels, top_n_words=5):
        """Extract representative documents and keywords for each cluster."""
        
        def extract_keywords(texts, top_n=5):
            """Extracts top keywords using TF-IDF."""
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Compute mean TF-IDF score per word
            tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top N keywords
            top_keywords = [feature_names[i] for i in np.argsort(tfidf_scores)[-top_n:]]
            return top_keywords
        
        
        cluster_docs = {cluster: [] for cluster in np.unique(labels)}
        
        for i, label in enumerate(labels):
            cluster_docs[label].append(texts[i])
        
        cluster_keywords = {}
        for cluster, docs in cluster_docs.items():
            # Extract keywords from the most representative documents
            keywords = extract_keywords(docs, top_n=top_n_words)
            cluster_keywords[cluster] = keywords
        
        return cluster_keywords
    
class SimilarityMetric():
    
    @staticmethod
    def compute_intra_cluster_similarity(trn_corpus, cluster_labels, centroids):
        """Computes the average cosine similarity between points and centroids for each cluster."""    
        intra_similarities = {}
        
        for idx, centroid in enumerate(centroids):
            cluster_indices = np.where(cluster_labels == idx)[0]
            cluster_corpus = trn_corpus[cluster_indices]
            
            # Compute cosine similarity of each point with the centroid
            similarities = cosine_similarity(cluster_corpus, centroid.reshape(1, -1)).flatten()
            
            # Store the mean similarity
            intra_similarities[idx] = np.mean(similarities)
        
        return intra_similarities
        
    @staticmethod
    def compute_inter_cluster_similarity(centroids):
        """Computes cosine similarity between cluster centroids to check if clusters are too similar."""
        
        len_centroids = len(centroids)
        
        # Compute pairwise cosine similarity between centroids
        similarity_matrix = cosine_similarity(centroids)
        
        # Store only upper triangle values (excluding diagonal)
        cluster_pairs = [(i, j, similarity_matrix[i, j]) for i in range(len_centroids) 
                         for j in range(i + 1, len_centroids)]

        return cluster_pairs