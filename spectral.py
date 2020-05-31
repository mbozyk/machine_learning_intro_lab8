from mpl_toolkits.mplot3d import Axes3D
import sklearn.decomposition as dc
import sklearn.datasets as sg
import sklearn.datasets as ds
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

X1, y1_true = sg.make_blobs(n_samples=300,centers=4, cluster_std=0.60, random_state=0)
X2, y2_true = ds.make_moons(200, noise=.05, random_state=0)
km1 = KMeans(n_clusters=4)
km1_pred = km1.fit_predict(X1, y1_true)
km2 = KMeans(n_clusters=2)
km2_pred = km2.fit_predict(X2, y2_true)


km1= KMeans(n_clusters=4).fit(X1)
plt.figure()
plt.scatter(X1[:, 0], X1[:, 1], c=km1.labels_)



km2= KMeans(n_clusters=2).fit(X2)
plt.figure()
plt.scatter(X2[:, 0], X2[:, 1], c=km2.labels_)


# plt.show()

clustering_1 = SpectralClustering(n_clusters=4, assign_labels='discretize', eigen_solver='arpack').fit(X1)
y1_spec = clustering_1.labels_

plt.figure()
plt.scatter(X1[:,0], X1[:,1], s=50, c=y1_spec)

plt.xlabel("X")
plt.ylabel("Y")


X2 = StandardScaler().fit_transform(X2)
clustering_2 = SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity='nearest_neighbors').fit(X2)
y2_spec = clustering_2.labels_.astype(np.int)
plt.figure()
plt.scatter(X2[:,0], X2[:,1], s=50, c=y2_spec)

plt.xlabel("X")
plt.ylabel("Y")

plt.show()
