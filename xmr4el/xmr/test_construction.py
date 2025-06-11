from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

def create_optimal_embeddings(texts):
    """Create the highest quality TF-IDF embeddings possible"""
    # Enhanced TF-IDF with careful parameter tuning
    tfidf = TfidfVectorizer(
        max_features=200000,  # Very large vocabulary
        min_df=2,            # Only include meaningful terms
        max_df=0.5,          # Filter out overly common terms
        ngram_range=(1, 3),  # Include unigrams, bigrams and trigrams
        analyzer='word',      # Word-level analysis
        sublinear_tf=True,    # Use log scaling
        use_idf=True,         # Use inverse document frequency
        smooth_idf=True,      # Smooth IDF weights
        lowercase=True,       # Case normalization
        stop_words='english'  # Remove stopwords
    )
    
    # Dimensionality reduction with careful whitening
    svd = TruncatedSVD(
        n_components=768,     # Match transformer dimensions
        algorithm='arpack',   # Most accurate SVD algorithm
        random_state=42
    )
    
    # L2 normalization
    normalizer = Normalizer(copy=False)
    
    # Create the pipeline
    pipeline = make_pipeline(
        tfidf,
        svd,
        normalizer
    )
    
    return pipeline.fit_transform(texts)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from tqdm import tqdm

class QualityDrivenHierarchicalClustering:
    def __init__(self, min_cluster_size=10, linkage_method='ward', affinity='euclidean'):
        self.min_cluster_size = min_cluster_size
        self.linkage_method = linkage_method
        self.affinity = affinity
        
    def _compute_optimal_k(self, embeddings):
        """Compute optimal k using multiple validation metrics"""
        max_k = min(50, len(embeddings) // self.min_cluster_size)
        if max_k < 2:
            return 1
            
        metrics = {
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
        
        # Evaluate multiple k values with full convergence
        for k in tqdm(range(2, max_k + 1), desc="Evaluating k values"):
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,  # More initializations for better stability
                max_iter=500,  # More iterations for full convergence
                algorithm='full',  # Classic EM algorithm
                random_state=42
            ).fit(embeddings)
            
            labels = kmeans.labels_
            
            # Only compute metrics if we have meaningful clusters
            if len(set(labels)) > 1:
                metrics['silhouette'].append(silhouette_score(embeddings, labels, metric='cosine'))
                metrics['calinski_harabasz'].append(calinski_harabasz_score(embeddings, labels))
                metrics['davies_bouldin'].append(davies_bouldin_score(embeddings, labels))
            else:
                metrics['silhouette'].append(-1)
                metrics['calinski_harabasz'].append(0)
                metrics['davies_bouldin'].append(float('inf'))
        
        # Combine metrics (higher is better)
        combined_scores = (
            0.5 * np.array(metrics['silhouette']) +
            0.3 * np.array(metrics['calinski_harabasz']) / max(metrics['calinski_harabasz']) +
            0.2 * (1 - np.array(metrics['davies_bouldin']) / max(metrics['davies_bouldin']))
        )
        
        return np.argmax(combined_scores) + 2  # Return best k (offset by 2)
    
    def _build_hierarchy(self, embeddings, depth=5):
        """Recursively build the hierarchy with optimal splits"""
        if depth == 0 or len(embeddings) <= self.min_cluster_size:
            return {'samples': embeddings, 'children': []}
        
        # Compute optimal k for this node
        k = self._compute_optimal_k(embeddings)
        
        # Perform high-quality clustering
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=20,  # Even more initializations
            max_iter=1000,
            algorithm='full',
            random_state=42
        ).fit(embeddings)
        
        # Build hierarchy recursively
        node = {'samples': embeddings, 'children': []}
        unique_labels = np.unique(kmeans.labels_)
        
        for label in unique_labels:
            cluster_mask = (kmeans.labels_ == label)
            cluster_embeddings = embeddings[cluster_mask]
            
            if len(cluster_embeddings) >= self.min_cluster_size:
                child_node = self._build_hierarchy(cluster_embeddings, depth-1)
                node['children'].append(child_node)
        
        return node
    
    def cluster(self, embeddings, depth=5):
        """Main clustering method"""
        # First perform global hierarchical clustering
        distance_matrix = pdist(embeddings, metric=self.affinity)
        linkage_matrix = linkage(
            distance_matrix,
            method=self.linkage_method,
            optimal_ordering=True
        )
        
        # Determine optimal cut
        best_cut = self._find_optimal_cut(linkage_matrix, distance_matrix)
        global_labels = fcluster(linkage_matrix, best_cut, criterion='distance')
        
        # Then refine each cluster with local hierarchical clustering
        final_clusters = []
        unique_labels = np.unique(global_labels)
        
        for label in unique_labels:
            cluster_embeddings = embeddings[global_labels == label]
            if len(cluster_embeddings) > self.min_cluster_size:
                subtree = self._build_hierarchy(cluster_embeddings, depth)
                final_clusters.append(subtree)
            else:
                final_clusters.append({'samples': cluster_embeddings, 'children': []})
        
        return final_clusters
    
    def _find_optimal_cut(self, linkage_matrix, distance_matrix):
        """Find the optimal cut point in the dendrogram"""
        # Compute inconsistency for each merge
        inconsistency = inconsistent(linkage_matrix)
        
        # Find the point with maximum inconsistency
        max_inc = np.argmax(inconsistency[:, 3])
        return linkage_matrix[max_inc, 2] * 1.1  # Slightly above the merge point
    
from sklearn.ensemble import VotingClassifier

class ClusteringEnsemble:
    def __init__(self, n_estimators=5):
        self.models = [
            QualityDrivenHierarchicalClustering() for _ in range(n_estimators)
        ]
        
    def fit_predict(self, embeddings):
        all_labels = []
        
        # Generate diverse clusterings
        for model in self.models:
            # Vary parameters slightly for diversity
            model.linkage_method = np.random.choice(['ward', 'average', 'complete'])
            model.affinity = np.random.choice(['euclidean', 'cosine'])
            labels = model.cluster(embeddings)
            all_labels.append(labels)
        
        # Use consensus clustering (simplified)
        return self._consensus_clustering(all_labels)
    
    def _consensus_clustering(self, all_labels):
        """Implement consensus clustering using co-association matrix"""
        # This would be a more sophisticated implementation in practice
        return np.mean(all_labels, axis=0).argmax(axis=1)
    
def optimize_clusters(embeddings, labels):
    """Refine clusters through iterative optimization"""
    improved = True
    while improved:
        improved = False
        centroids = [embeddings[labels == i].mean(axis=0) for i in np.unique(labels)]
        
        # Reassign points to nearest centroid
        new_labels = np.argmin(
            np.array([[np.linalg.norm(x - c) for c in centroids] for x in embeddings]),
            axis=1
        )
        
        # Check for improvement
        if not np.array_equal(labels, new_labels):
            improved = True
            labels = new_labels
            
        # Remove empty clusters
        unique_labels = np.unique(labels)
        labels = np.searchsorted(unique_labels, labels)
        
    return labels

class ClusterEvaluator:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        
    def evaluate(self, labels):
        metrics = {}
        
        if len(np.unique(labels)) > 1:
            metrics['silhouette'] = silhouette_score(self.embeddings, labels, metric='cosine')
            metrics['calinski_harabasz'] = calinski_harabasz_score(self.embeddings, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(self.embeddings, labels)
            
            # Cluster separation metrics
            metrics['separation'] = self._cluster_separation(labels)
            metrics['cohesion'] = self._cluster_cohesion(labels)
            metrics['dunn_index'] = self._dunn_index(labels)
            
        return metrics
    
    def _cluster_separation(self, labels):
        """Measure how well separated clusters are"""
        centroids = []
        for label in np.unique(labels):
            centroids.append(self.embeddings[labels == label].mean(axis=0))
        return np.min(pdist(centroids, 'cosine'))
    
    def _cluster_cohesion(self, labels):
        """Measure how tight clusters are"""
        cohesion = 0
        for label in np.unique(labels):
            cluster_points = self.embeddings[labels == label]
            centroid = cluster_points.mean(axis=0)
            cohesion += np.mean([cosine_similarity([x], [centroid])[0][0] for x in cluster_points])
        return cohesion / len(np.unique(labels))
    
    def _dunn_index(self, labels):
        """Dunn index for cluster validation"""
        # Implementation would calculate inter-cluster distances vs intra-cluster distances
        pass
    
# Consider adding these to your embedding pipeline
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE

def enhance_embeddings(embeddings):
    # Non-linear dimensionality enhancement
    tsne = TSNE(n_components=0.5*embeddings.shape[1], perplexity=50, method='exact')
    tsne_emb = tsne.fit_transform(embeddings)
    
    # Random projections for additional views
    grp = GaussianRandomProjection(n_components=0.3*embeddings.shape[1])
    grp_emb = grp.fit_transform(embeddings)
    
    return np.hstack([embeddings, tsne_emb, grp_emb])