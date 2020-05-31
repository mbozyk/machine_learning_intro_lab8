import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
np.random.seed(12345)
no_points_per_set = 30

x1 = np.random.randn(no_points_per_set, 3) + np.array([5, -3, 7])
x2 = np.random.randn(no_points_per_set, 3) + np.array([2, -5, 9])
x3 = np.random.randn(no_points_per_set, 3) + np.array([7, 5, 4])



X = np.concatenate((x1, x2, x3))
y = np.zeros(X.shape[0])
y[30:60] = 1
y[60:] = 2

pca1 = PCA(n_components=1)
model1 = pca1.fit(X)
prinComp_1 = model1.transform(X)

plt.figure()
inv_prinComp_1 = pca1.inverse_transform(prinComp_1)
plt.plot(inv_prinComp_1[:,0][y==0],inv_prinComp_1[:,1][y==0], 'r*')
plt.plot(inv_prinComp_1[:,0][y==1],inv_prinComp_1[:,1][y==1], 'b*')
plt.plot(inv_prinComp_1[:,0][y==2],inv_prinComp_1[:,1][y==2], 'g*')
plt.axis('equal')



pca2 = PCA(n_components=2)
model2 = pca2.fit(X)
prinComp_2 = model2.transform(X)
plt.figure()
plt.plot(prinComp_2[:,0][y == 0], prinComp_2[:,1][y == 0], "r*")
plt.plot(prinComp_2[:,0][y == 1], prinComp_2[:,1][y == 1], "b*")
plt.plot(prinComp_2[:,0][y == 2], prinComp_2[:,1][y == 2], "g*")


fig = plt.figure(dpi=100)
ax = fig.add_subplot(111, projection= '3d')
plt.plot(X[:,0][y == 0], X[:,1][y == 0], X[:,2][y == 0], "r*")
plt.plot(X[:,0][y == 1], X[:,1][y == 1], X[:,2][y == 1], "b*")
plt.plot(X[:,0][y == 2], X[:,1][y == 2], X[:,2][y == 2], "g*")
plt.show()
print(model1.explained_variance_ratio_.sum())
print(model2.explained_variance_ratio_.sum())