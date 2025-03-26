from typing import Counter
from src.xmr.xmr_tree import XMRTree

htree = XMRTree.load()

labels_3_3 = htree.children[3].clustering_model.model.labels_

print(Counter(labels_3_3))

print(htree)
