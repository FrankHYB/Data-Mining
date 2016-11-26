import numpy as np
import operator
import sys
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

faces = fetch_olivetti_faces()
samples,h,w = faces.images.shape
feature = faces.data
feature_size = feature.shape[1]

label = faces.target
label_name = np.unique(label)
class_size = label_name.shape

print("Number of samples: %d" % samples)
print("Number of features: %d" % feature_size)
print("Number of classes: %d" % class_size)


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


def softmax(matrix):
    numerator = np.exp(matrix - np.max(matrix))
    ans = numerator / numerator.sum()
    return ans


def softmaxPredict(matrix,testFeature):
    """ Compute the class probabilities for each example """

    #matrix = matrix.reshape(class_size, testFeature.shape)
    dot = np.dot(matrix, testFeature)
    numerator = np.exp(dot)
    probabilities = numerator / np.sum(numerator, axis=0)

    predictions = np.zeros((testFeature.shape[1], 1))
    predictions[:, 0] = np.argmax(probabilities, axis=0)

    return predictions



trainingFeature, testFeature, trainingLabel, testLabel = train_test_split(
    feature, label, test_size=0.5, random_state=42)

num_of_train = 20
print("Extracting the top %d eigenfaces from %d faces"
      % (num_of_train, trainingFeature.shape[0]))

reducedMatrix = PCA(trainingFeature)
softmax_regressor = softmax(reducedMatrix)
#print Softmax_regressor

# TO DO: prediction
print softmax_regressor.shape
print testFeature.shape
#predict_softmax = softmaxPredict(softmax_regressor,testFeature)
#print predict_softmax


