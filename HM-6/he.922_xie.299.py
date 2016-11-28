import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import sys
import random
import math


def PCA(matrix, K ,num = 64):
    #matrix = np.delete(matrix, np.arange(2048, 4096), 1)#currently 2048 for performance
    #cov_mat = np.cov([row for row in matrix.T])
    avgMatrix = np.mean(matrix, 0)
    for i in range(matrix.shape[0]):
        matrix[i, :] = matrix[i, :] - avgMatrix
    eigen_val,eigen_vec = np.linalg.eig(matrix.T.dot(matrix))
    eigen_pairs = [(eigen_val[i], eigen_vec[:, i]) for i in range(len(eigen_val))]
    eigen_pairs.sort(key=lambda x:x[0], reverse=True)
    columns = []
    for i in range(num):
         columns.append(eigen_pairs[i][1].reshape(K, 1))
    matrix_w = np.hstack(tuple(columns))

    return matrix_w.T.dot(matrix.T)
    #print selected_eigen_pairs(eigen_pairs)[0][1].reshape(64, 4096)


def compute_softmax(trainFeature, trainLable, w, K):
    L = 1e-6
    N,D = trainFeature.shape
    w = w.reshape((K,D))

    xw = np.dot(trainFeature,np.transpose(w))
    #Avoid overflow
    xw -= np.tile(xw.max(axis=1).reshape((N,1)),(1,K))
    expXW = np.exp(xw)
    sumExpXW = expXW.sum(axis=1)
    XWy = xw[range(N),trainLable]

    #compute gradients
    gradient = np.zeros((K, D))
    for k in range(K):
        indk = np.where(trainLable == k)[0]
        gradient[k, :] = -1.0 / N * trainFeature[indk, :].sum(axis=0).reshape((D,)) \
                  + 1.0 / N * np.dot(expXW[:, k] / sumExpXW, trainFeature).reshape((D,)) + L*w[k,:].reshape((D,))

    grad = gradient.reshape((K * D,))
    return grad



def softmax_predict(w, feature, K):
    N,D = feature.shape
    w = w.reshape((K,D))
    prediction = np.argmax(np.dot(feature,np.transpose(w)),axis=1)
    return prediction


def test_softmax(predictions,testLabel):
    correct_index = np.where(predictions == testLabel)[0]
    accuracy = float(correct_index.size) / float(predictions.size)
    print 'softmax accuracy = ', str(accuracy)



def init_weight(D,K):
    w = np.zeros((D*K,))
    return w

def svm(trainFeatures, trainLabels, testFeatures, testLabels):
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],}
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid).fit(trainFeatures, trainLabels)
    predict = clf.predict(testFeatures)
    print predict
    print testLabels
    correct_index = np.where(predict == testLabels)[0]
    print correct_index

    print 'svm accuracy = ' + str(float(correct_index.size) / float(predict.size))

def knn(trainingFeatures, testFeatures, testLabels, trainingLabels ,K = 300):
    predict = []
    for i in range(testFeatures.shape[0]):
        dist = sys.maxint
        prediction = -1
        for j in range(trainingFeatures.shape[0]):
            res = np.linalg.norm(trainingFeatures[j, :]- testFeatures[i, :].T)
            if res < dist:
                dist = res
                prediction = trainingLabels[j]
        predict.append(prediction)

    true_num = 0
    print predict
    print testLabels
    for i in range(len(predict)):
        if predict[i] == testLabels[i]:
            true_num += 1

    print float(true_num) / float(testLabels.shape[0])






if __name__ == "__main__":

    faces = fetch_olivetti_faces()
    samples, h, w = faces.images.shape
    feature = faces.data
    feature_size = feature.shape[1]

    label = faces.target
    label_name = np.unique(label)
    class_size = label_name.shape

    #print("Number of samples: %d" % samples)
    #print("Number of features: %d" % feature_size)
    #print("Number of classes: %d" % class_size)

    trainingFeature, testFeature, trainingLabel, testLabel = train_test_split(
    feature, label, test_size=0.25, random_state=None)

    num_of_train = trainingFeature.shape[0]
    print("Extracting the top %d eigenfaces from %d faces"
      % (num_of_train, 400))


    print "Test lable"
    print testLabel.shape
    print trainingLabel.shape

#reducedMatrix = PCA(trainingFeature)
#reducedMatrix = trainingFeature
#testMatrix = testFeature
#testMatrix = PCA(testFeature)
#for i in range(10):
 #   softmax_regressor = softmax(reducedMatrix)
#print Softmax_regressor
    K,D = trainingFeature.shape
    print 'K= ' + str(K) + ' D = ' + str(D)
    iterations = 1000
    naught = 10.0

    #initial weight
    w = init_weight(D,K)
    w = np.random.normal(size=w.size)


    #for i in range(1,iterations):
        #Randomly choose 10 samples
        #index = np.random.choice(K,size =(10),replace=False)
        #x = trainingFeature[index,:]
        #y = trainingLabel[index]
        #grad= compute_softmax(x,y,w,K)

        #update weight
        #w -= naught/np.sqrt(len(index)*i)*grad

    #preds = softmax_predict(w,testFeature,K)
    #accuracy = test_softmax(preds,testLabel)

    trainingFeature = PCA(trainingFeature, 4096, 3072)
    testFeature = PCA(testFeature, 4096, 3072)
    #svm(trainingFeature.T, trainingLabel, testFeature.T, testLabel)
    knn(trainingFeature.T, testFeature.T, testLabel, trainingLabel)

