# -*- coding: utf-8 -*-

from pprint import pprint
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


# =========================================================
# data
# =========================================================
centers = [[1, 1],
		   [-1, -1],
		   [1, -1]]
X, labels_true = make_blobs(n_samples = 750,
							centers = centers,
							cluster_std = 0.4,
							random_state = 0)
X = StandardScaler().fit_transform(X)


# =========================================================
# clustering
# =========================================================
db = DBSCAN(eps = 0.3, min_samples = 10)
db.fit(X)
pprint(db)
pprint(db.labels_)
pprint(db.core_sample_indices_)

core_sample_mask = np.zeros_like(db.labels_, dtype = bool)
core_sample_mask[db.core_sample_indices_] = True
labels = db.labels_
print(core_sample_mask)



