#!/usr/bin/python
# -*- coding: UTF-8 -*-
# by honghu in 2020.3.6

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
from sklearn import datasets, manifold
from sklearn.datasets import make_swiss_roll


def get_distance_matrix(data):
	expand_ = data[:, np.newaxis, :]
	repeat1 = np.repeat(expand_, data.shape[0], axis=1)
	repeat2 = np.swapaxes(repeat1, 0, 1)
	D = np.linalg.norm(repeat1 - repeat2, ord=2, axis=-1, keepdims=True).squeeze(-1)
	return D

def get_matrix_B(D):
	assert D.shape[0] == D.shape[1]
	DD = np.square(D)
	sum_ = np.sum(DD, axis=1) / D.shape[0]
	Di = np.repeat(sum_[:, np.newaxis], D.shape[0], axis=1)
	Dj = np.repeat(sum_[np.newaxis, :], D.shape[0], axis=0)
	Dij = np.sum(DD) / ((D.shape[0])**2) * np.ones([D.shape[0], D.shape[0]])
	B = (Di + Dj - DD- Dij) / 2
	return B

def MDS(data, n=2):
	D = get_distance_matrix(data)
	B = get_matrix_B(D)
	B_value, B_vector = np.linalg.eigh(B)
	Be_sort = np.argsort(-B_value)
	B_value = B_value[Be_sort]               # 降序排列的特征值
	B_vector = B_vector[:,Be_sort]           # 降序排列的特征值对应的特征向量
	Bez = np.diag(B_value[0:n])
	Bvz = B_vector[:, 0:n]
	Z = np.dot(np.sqrt(Bez), Bvz.T).T
	return Z

def test_iris():
	iris = datasets.load_iris()
	data = iris.data                         # [150, 4]
	target = iris.target                     # [150]
	# print(np.shape(data))
	Z = MDS(data)
	figure1 = plt.figure()
	plt.subplot(1, 2, 1)
	plt.plot(Z[target==0, 0], Z[target==0, 1], 'r*', markersize=20)
	plt.plot(Z[target==1, 0], Z[target==1, 1], 'bo', markersize=20)
	plt.plot(Z[target==2, 0], Z[target==2, 1], 'gx', markersize=20)
	plt.title('CUSTOM')
	plt.subplot(1, 2, 2)
	Z1 = manifold.MDS(n_components=2).fit_transform(data)
	plt.plot(Z1[target==0,0], Z1[target==0,1], 'r*', markersize=20)
	plt.plot(Z1[target==1,0], Z1[target==1,1], 'bo', markersize=20)
	plt.plot(Z1[target==2,0], Z1[target==2,1], 'gx', markersize=20)
	plt.title('SKLEARN')
	plt.show()

def test_swiss_roll(n_samples=5000):
	X, t = make_swiss_roll(n_samples=n_samples, noise=0.2, random_state=42) # X为坐标 t为颜色
	Z = MDS(X)
	# figure1=plt.figure()
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
	# test_iris()
	test_swiss_roll()
