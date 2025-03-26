from typing import Counter
from src.models.cluster_wrapper.clustering_model import ClusteringModel
from src.xmr.xmr_tree import XMRTree

xtree = XMRTree.load()

# clustering_config = {'type': 'sklearnkmeans', 'kwargs': {}}
# clustering_config['kwargs'] = xtree.clustering_model.config

# xtree.clustering_model = ClusteringModel(clustering_config, xtree.clustering_model)

# xtree.save("data/saved_trees")
