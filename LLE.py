import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
from sklearn import datasets, manifold
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
from mpl_toolkits.mplot3d import Axes3D


def cal_pairwise_dist(data):
	expand_ = data[:, np.newaxis, :]
	repeat1 = np.repeat(expand_, data.shape[0], axis=1)
	repeat2 = np.swapaxes(repeat1, 0, 1)
	D = np.linalg.norm(repeat1 - repeat2, ord=2, axis=-1, keepdims=True).squeeze(-1)
	return D


def get_n_neighbors(data, n_neighbors=10):
	dist = cal_pairwise_dist(data)
	dist[dist < 0] = 0
	n = dist.shape[0]
	N = np.zeros((n, n_neighbors))
	for i in range(n):
		# np.argsort 列表从小到大的索引
		index_ = np.argsort(dist[i])[1:n_neighbors+1]
		N[i] = N[i] + index_
	return N.astype(np.int32)                         # [n_features, n_neighbors]


def lle(data, n_dims=2, n_neighbors=10):
	N = get_n_neighbors(data, n_neighbors)            # k近邻索引
	n, D = data.shape                                 # n_samples, n_features
	# prevent Si to small
	if n_neighbors > D:
		tol = 1e-3
	else:
		tol = 0
	# calculate W
	W = np.zeros((n_neighbors, n))
	I = np.ones((n_neighbors, 1))
	for i in range(n):                                # data[i] => [1, n_features]
		Xi = np.tile(data[i], (n_neighbors, 1)).T     # [n_features, n_neighbors]
		                                              # N[i] => [1, n_neighbors]
		Ni = data[N[i]].T                             # [n_features, n_neighbors]
		Si = np.dot((Xi-Ni).T, (Xi-Ni))               # [n_neighbors, n_neighbors]
		Si = Si + np.eye(n_neighbors)*tol*np.trace(Si)
		Si_inv = np.linalg.pinv(Si)
		wi = (np.dot(Si_inv, I)) / (np.dot(np.dot(I.T, Si_inv), I)[0,0])
		W[:, i] = wi[:,0]
	# print("Xi.shape", Xi.shape)
	# print("Ni.shape", Ni.shape)
	# print("Si.shape", Si.shape)
	print("W.shape", W.shape)
	W_y = np.zeros((n, n))
	for i in range(n):
		index = N[i]
		for j in range(n_neighbors):
			W_y[index[j],i] = W[j,i]
	I_y = np.eye(n)
	M = np.dot((I_y - W_y), (I_y - W_y).T)
	eig_val, eig_vector = np.linalg.eig(M)
	index_ = np.argsort(np.abs(eig_val))[1:n_dims+1]
	print("index_", index_)
	Y = eig_vector[:, index_]
	print("eig_vector.shape", eig_vector.shape)
	print("Y.shape", Y.shape)
	return Y


def test_swiss_roll(n_samples=2000):
	X, t = make_swiss_roll(n_samples=n_samples, noise=0.2, random_state=42) # X为坐标 t为颜色
	Z = lle(X)
	axes = [-11.5, 14, -2, 23, -12, 15]	
	fig = plt.figure()
	ax = fig.add_subplot(121, projection='3d')
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
	ax.view_init(10, 60)
	ax.set_xlabel("$x$", fontsize=18)
	ax.set_ylabel("$y$", fontsize=18)
	ax.set_zlabel("$z$", fontsize=18)
	ax.set_xlim(axes[0:2])
	ax.set_ylim(axes[2:4])
	ax.set_zlim(axes[4:6])
	plt.title('3D swiss roll')
	ax2 = fig.add_subplot(122)
	ax2.scatter(Z[:, 0], Z[:, 1], c=t, cmap=plt.cm.hot)
	ax2.set_xlabel("$x$", fontsize=18)
	ax2.set_ylabel("$y$", fontsize=18)
	plt.title('after MDS')
	plt.show()


if __name__ == '__main__':
	test_swiss_roll(2000)
