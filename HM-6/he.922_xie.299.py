import numpy as np
import operator
import sys
from sklearn.datasets import fetch_olivetti_faces
data = fetch_olivetti_faces()
targets = data.target


def PCA(matrix, num = 64):
    matrix = np.delete(matrix, np.arange(2048, 4096), 1)#currently 2048 for performance
    cov_mat = np.cov([row for row in matrix.T])
    eigen_val,eigen_vec = np.linalg.eig(cov_mat)
    eigen_pairs = [(eigen_val[i], eigen_vec[:, i]) for i in range(len(eigen_val))]
    eigen_pairs.sort(key=lambda x:x[0], reverse=True)
    columns = []
    for i in range(num):
         columns.append(eigen_pairs[i][1].reshape(2048, 1))
    matrix_w = np.hstack(tuple(columns))

    return matrix_w.T.dot(matrix.T)
    #print selected_eigen_pairs(eigen_pairs)[0][1].reshape(64, 4096)




print PCA(data.data)

