import numpy as np

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


"""
    Key Features:
        - Linkage Criteria: Single, Complete, Average, Ward's;
        - Distance Matrics: Euclidean, Manhattan, Cosine, etc;
        - Data PreProcessing: Scaling, Dimensionality reduction, handling missing data;
        - Tree Representation
        
        
    given a dataset (d1, d2, d3, ....dN) of size N
    at the top we have all data in one cluster
    the cluster is split using a flat clustering method eg. K-Means etc
    repeat
    choose the best cluster among all the clusters to split
    split that cluster by the flat clustering algorithm
    until each data is in its own singleton cluster   
    
    Utilizar K-Means Clustering do cuml como modelo para Hierarchical Clustering GPU - Divise Approach  
""" 

"""            
    Pecos params:
            
    nr_splits (int, optional): The out-degree of each internal node of the tree. Default is `16`.
            
    min_codes (int): The number of direct child nodes that the top level of the hierarchy should have.
            
    max_leaf_size (int, optional): The maximum size of each leaf node of the tree. Default is `100`.
            
    spherical (bool, optional): True will l2-normalize the centroids of k-means after each iteration. Default is `True`.
            
    seed (int, optional): Random seed. Default is `0`.
            
    kmeans_max_iter (int, optional): Maximum number of iterations for each k-means problem. Default is `20`.
            
    threads (int, optional): Number of threads to use. `-1` denotes all CPUs. Default is `-1`.
            
    do_sample (bool, optional): Do sampling if is True. Default is False.
    We use linear sampling strategy with warmup, which linearly increases sampling rate from `min_sample_rate` to `max_sample_rate`.
    The top (total_layer * `warmup_ratio`) layers are warmup_layers which use a fixed sampling rate `min_sample_rate`.
    The sampling rate for layer l is `min_sample_rate`+max(l+1-warmup_layer,0)*(`max_sample_rate`-min_sample_rate)/(total_layers-warmup_layers).
    Please refer to 'self.get_layer_sample_rate()' function for complete definition.
            
    max_sample_rate (float, optional): the maximum samplng rate at the end of the linear sampling strategy. Default is `1.0`.
            
    min_sample_rate (float, optional): the minimum sampling rate at the begining warmup stage of the linear sampling strategy. Default is `0.1`.
    Note that 0 < min_sample_rate <= max_sample_rate <= 1.0.
            
    warmup_ratio: (float, optional): The ratio of warmup layers. 0 <= warmup_ratio <= 1.0. Default is 0.4.
"""

class DivisiveHierarchicalClustering:
    
    def __init__(self):
        pass    
    
    

class KmeansRanker():
    
    def __init__(self):
        pass
    
    """
        kmeans
    
        [{'clusters': np.int64(2), 'silhouette_avg': np.float64(0.0012874925288218614)}, 
        {'clusters': np.int64(3), 'silhouette_avg': np.float64(0.001645716899949783)}, 
        {'clusters': np.int64(4), 'silhouette_avg': np.float64(0.0020764704365701294)},
        {'clusters': np.int64(5), 'silhouette_avg': np.float64(0.0025912438745330124)},
        {'clusters': np.int64(6), 'silhouette_avg': np.float64(0.002588087303212956)}, 
        {'clusters': np.int64(7), 'silhouette_avg': np.float64(0.0028781926001339017)},
        {'clusters': np.int64(8), 'silhouette_avg': np.float64(0.0031168476544459725)}, 
        {'clusters': np.int64(9), 'silhouette_avg': np.float64(0.0035738710844732246)}, 
        {'clusters': np.int64(10), 'silhouette_avg': np.float64(0.0037454451038614624)}, 
        {'clusters': np.int64(11), 'silhouette_avg': np.float64(0.004182600335483569)}, 
        {'clusters': np.int64(12), 'silhouette_avg': np.float64(0.0040986804954275926)}, 
        {'clusters': np.int64(13), 'silhouette_avg': np.float64(0.004408731861059765)}, 
        {'clusters': np.int64(14), 'silhouette_avg': np.float64(0.004876728435872956)}]
    
        mini_batch_kmeans
        
        [{'clusters': np.int64(8), 'silhouette_avg': np.float64(0.00310304490478474)}, 
        {'clusters': np.int64(9), 'silhouette_avg': np.float64(0.002848751004613504)}, 
        {'clusters': np.int64(10), 'silhouette_avg': np.float64(0.003696486765092457)}, 
        {'clusters': np.int64(11), 'silhouette_avg': np.float64(0.0017501373288484846)}, 
        {'clusters': np.int64(12), 'silhouette_avg': np.float64(0.0028933034769265532)}, 
        {'clusters': np.int64(13), 'silhouette_avg': np.float64(0.003075694909461008)}, 
        {'clusters': np.int64(14), 'silhouette_avg': np.float64(0.00486279844326867)}, 
        {'clusters': np.int64(15), 'silhouette_avg': np.float64(0.004874414924533095)}, 
        {'clusters': np.int64(16), 'silhouette_avg': np.float64(0.003728346421939408)}, 
        {'clusters': np.int64(17), 'silhouette_avg': np.float64(0.0016602247181069134)}, 
        {'clusters': np.int64(18), 'silhouette_avg': np.float64(0.004432539919444544)}, 
        
        {'clusters': np.int64(19), 'silhouette_avg': np.float64(0.004676646721967078)}]
        
        normalized mini_batch_kmeans, random state = 0
    
        [{'clusters': np.int64(2), 'silhouette_avg': np.float64(0.0017411843114807607)}, 
        {'clusters': np.int64(3), 'silhouette_avg': np.float64(0.0015132508385282601)}, 
        {'clusters': np.int64(4), 'silhouette_avg': np.float64(0.002098291330511073)}, 
        {'clusters': np.int64(5), 'silhouette_avg': np.float64(0.002205358319696167)}, 
        {'clusters': np.int64(6), 'silhouette_avg': np.float64(0.0025236294727478785)}, 
        {'clusters': np.int64(7), 'silhouette_avg': np.float64(0.002761132316100299)}, 
        {'clusters': np.int64(8), 'silhouette_avg': np.float64(0.0017996632371357819)}, 
        {'clusters': np.int64(9), 'silhouette_avg': np.float64(0.0023175143304846367)}, 
        {'clusters': np.int64(10), 'silhouette_avg': np.float64(0.0014894578477099149)}, 
        {'clusters': np.int64(11), 'silhouette_avg': np.float64(0.0010668498696209125)}, 
        {'clusters': np.int64(12), 'silhouette_avg': np.float64(0.00029784989620512903)}, 
        {'clusters': np.int64(13), 'silhouette_avg': np.float64(0.0023491912029062053)}, 
        {'clusters': np.int64(14), 'silhouette_avg': np.float64(0.001090077419883189)}, 
        {'clusters': np.int64(15), 'silhouette_avg': np.float64(0.002223352160551781)}, 
        {'clusters': np.int64(16), 'silhouette_avg': np.float64(0.003453439426442321)}, 
        {'clusters': np.int64(17), 'silhouette_avg': np.float64(0.0017192259303251093)}, 
        {'clusters': np.int64(18), 'silhouette_avg': np.float64(0.0030982060857467795)}, 
        {'clusters': np.int64(19), 'silhouette_avg': np.float64(0.003013625735436723)}, 
        {'clusters': np.int64(20), 'silhouette_avg': np.float64(0.0022120490730275226)}, 
        {'clusters': np.int64(21), 'silhouette_avg': np.float64(0.0038010239316620463)}, 
        {'clusters': np.int64(22), 'silhouette_avg': np.float64(0.003770523830651899)}, 
        {'clusters': np.int64(23), 'silhouette_avg': np.float64(0.0016205724637064008)}, 
        BEST {'clusters': np.int64(24), 'silhouette_avg': np.float64(0.004969569595632096)}, 
        {'clusters': np.int64(25), 'silhouette_avg': np.float64(0.0013149857848656437)}, 
        {'clusters': np.int64(26), 'silhouette_avg': np.float64(0.003294369445542648)}, 
        {'clusters': np.int64(27), 'silhouette_avg': np.float64(0.003198002181322837)}, 
        {'clusters': np.int64(28), 'silhouette_avg': np.float64(0.0024932757824032248)}, 
        {'clusters': np.int64(29), 'silhouette_avg': np.float64(0.000795446856895327)}]
        
        [{'clusters': np.int64(30), 'silhouette_avg': np.float64(0.0009744340819551073)}, 
        {'clusters': np.int64(31), 'silhouette_avg': np.float64(0.00027662916720317115)}, 
        {'clusters': np.int64(32), 'silhouette_avg': np.float64(0.002715292155713557)}, 
        {'clusters': np.int64(33), 'silhouette_avg': np.float64(0.0018380244259579553)}, 
        {'clusters': np.int64(34), 'silhouette_avg': np.float64(0.0016274407566855146)}, 
        {'clusters': np.int64(35), 'silhouette_avg': np.float64(0.0021408652803022116)}, 
        {'clusters': np.int64(36), 'silhouette_avg': np.float64(0.0011847811311640841)}, 
        {'clusters': np.int64(37), 'silhouette_avg': np.float64(0.0008128983085119447)}, 
        {'clusters': np.int64(38), 'silhouette_avg': np.float64(-0.0007893953437476029)}, 
        {'clusters': np.int64(39), 'silhouette_avg': np.float64(-0.0027702346810444837)}, 
        {'clusters': np.int64(40), 'silhouette_avg': np.float64(-0.00041068789470155375)}, 
        {'clusters': np.int64(41), 'silhouette_avg': np.float64(-0.0007024811673212377)}, 
        {'clusters': np.int64(42), 'silhouette_avg': np.float64(-0.0006014973896600702)}, 
        {'clusters': np.int64(43), 'silhouette_avg': np.float64(-0.0016399093683058857)}, 
        {'clusters': np.int64(44), 'silhouette_avg': np.float64(-0.00020859890268555673)}, 
        {'clusters': np.int64(45), 'silhouette_avg': np.float64(-0.0014252955208222636)}, 
        {'clusters': np.int64(46), 'silhouette_avg': np.float64(0.0013721377529976734)}, 
        {'clusters': np.int64(47), 'silhouette_avg': np.float64(-0.0009028715854021754)}, 
        {'clusters': np.int64(48), 'silhouette_avg': np.float64(-0.001427868498065002)}, 
        {'clusters': np.int64(49), 'silhouette_avg': np.float64(-0.0038864460653717805)}]
        
        [{'clusters': np.int64(50), 'silhouette_avg': np.float64(-0.001844888848171422)}, 
        {'clusters': np.int64(51), 'silhouette_avg': np.float64(-0.0005784658962918869)}, 
        {'clusters': np.int64(52), 'silhouette_avg': np.float64(0.0011589327694288025)}, 
        {'clusters': np.int64(53), 'silhouette_avg': np.float64(-0.00306072964498252)}, 
        {'clusters': np.int64(54), 'silhouette_avg': np.float64(-0.0006125002197093724)}, 
        {'clusters': np.int64(55), 'silhouette_avg': np.float64(-0.002116876540804838)}, 
        {'clusters': np.int64(56), 'silhouette_avg': np.float64(-0.00037534646000190097)}, 
        {'clusters': np.int64(57), 'silhouette_avg': np.float64(-0.0006609446917386185)}, 
        {'clusters': np.int64(58), 'silhouette_avg': np.float64(-0.0006431271985275712)}, 
        {'clusters': np.int64(59), 'silhouette_avg': np.float64(-0.0025408808122046496)}, 
        {'clusters': np.int64(60), 'silhouette_avg': np.float64(-0.00031938766854774095)}, 
        {'clusters': np.int64(61), 'silhouette_avg': np.float64(-0.002583279675383354)}, 
        {'clusters': np.int64(62), 'silhouette_avg': np.float64(-0.003452254313936105)}, 
        {'clusters': np.int64(63), 'silhouette_avg': np.float64(-0.004049703467681716)}, 
        {'clusters': np.int64(64), 'silhouette_avg': np.float64(-0.0012419406853118383)},
        {'clusters': np.int64(65), 'silhouette_avg': np.float64(-0.004889728803866899)}, 
        {'clusters': np.int64(66), 'silhouette_avg': np.float64(0.0010237486897842335)}, 
        {'clusters': np.int64(67), 'silhouette_avg': np.float64(-0.002872963158410539)}, 
        {'clusters': np.int64(68), 'silhouette_avg': np.float64(-0.004913654394743954)}, 
        {'clusters': np.int64(69), 'silhouette_avg': np.float64(0.00313772784817452)}, 
        {'clusters': np.int64(70), 'silhouette_avg': np.float64(-0.0013662612661001642)}, 
        {'clusters': np.int64(71), 'silhouette_avg': np.float64(-0.0037937759616261243)}, 
        {'clusters': np.int64(72), 'silhouette_avg': np.float64(-0.007429972116878059)}, 
        {'clusters': np.int64(73), 'silhouette_avg': np.float64(-0.004554557244377862)}, 
        {'clusters': np.int64(74), 'silhouette_avg': np.float64(-0.004198503262961135)}, 
        {'clusters': np.int64(75), 'silhouette_avg': np.float64(-0.0034619501924973654)}, 
        {'clusters': np.int64(76), 'silhouette_avg': np.float64(-0.002887759006525963)}, 
        {'clusters': np.int64(77), 'silhouette_avg': np.float64(-0.001390641926715123)}, 
        {'clusters': np.int64(78), 'silhouette_avg': np.float64(-0.00822039414518552)}, 
        {'clusters': np.int64(79), 'silhouette_avg': np.float64(-0.004879107580409058)}, 
        {'clusters': np.int64(80), 'silhouette_avg': np.float64(-0.009641644979860687)}, 
        {'clusters': np.int64(81), 'silhouette_avg': np.float64(-0.005155265673483351)}, 
        {'clusters': np.int64(82), 'silhouette_avg': np.float64(-0.005732457920695442)}, 
        {'clusters': np.int64(83), 'silhouette_avg': np.float64(-0.004625724668847459)}, 
        {'clusters': np.int64(84), 'silhouette_avg': np.float64(0.0002473117828995038)}, 
        {'clusters': np.int64(85), 'silhouette_avg': np.float64(-0.006935825197237461)}, 
        {'clusters': np.int64(86), 'silhouette_avg': np.float64(-0.007363918832000625)}, 
        {'clusters': np.int64(87), 'silhouette_avg': np.float64(-0.01483566646444627)}, 
        {'clusters': np.int64(88), 'silhouette_avg': np.float64(-0.008320504576605206)}, 
        {'clusters': np.int64(89), 'silhouette_avg': np.float64(-0.007058832206769777)}, 
        {'clusters': np.int64(90), 'silhouette_avg': np.float64(-0.010800533185690764)}, 
        {'clusters': np.int64(91), 'silhouette_avg': np.float64(-0.012100670311745458)}, 
        {'clusters': np.int64(92), 'silhouette_avg': np.float64(-0.007535965527524457)}, 
        {'clusters': np.int64(93), 'silhouette_avg': np.float64(-0.008084524255977662)}, 
        {'clusters': np.int64(94), 'silhouette_avg': np.float64(-0.011432205662566265)}, 
        {'clusters': np.int64(95), 'silhouette_avg': np.float64(-0.009096183490451276)}, 
        {'clusters': np.int64(96), 'silhouette_avg': np.float64(-0.009959800260093434)}, 
        {'clusters': np.int64(97), 'silhouette_avg': np.float64(-0.01094391462209544)}, 
        {'clusters': np.int64(98), 'silhouette_avg': np.float64(-0.011932858391592062)}, 
        {'clusters': np.int64(99), 'silhouette_avg': np.float64(-0.013502470628765315)}]
    
    """
    
    
    def ranker(self, X):
        
        scores = []
        
        X_normalized = normalize(X, norm='l2')
        
        for i in np.arange(2, 10):
            
            # kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
            kmeans = MiniBatchKMeans(n_clusters=i, random_state=0).fit(X_normalized)
            kmeans_labels = kmeans.labels_
            
            silhouette_avg = silhouette_score(X_normalized, kmeans_labels)
            
            scores.append({'clusters':i, 
                           'silhouette_avg': silhouette_avg})
            
            print("Ran KMeans", i)
            
        return scores
            
            
        
# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    seed = np.random.seed(42)
    X, _ = make_blobs(n_samples=200, centers=4, n_features=10, random_state=42) # 50 samples with 2 features
    
    # Create and fit the Divisive Hierarchical K-Means model
    dhkmeans = DivisiveHierarchicalClustering(max_leaf_size=50)
    
    #n_splits=16, min_leaf_size=20, max_leaf_size=100, spherical=True, seed=0, kmeans_max_iter=20
    clusters = dhkmeans.fit(X)
    
    print(clusters)

    
        