import numpy as np
from heapq import heappush, heappop
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import operator


def pca(matrix, K, avgMatrix ,num = 64):
    #matrix = np.delete(matrix, np.arange(2048, 4096), 1)#currently 2048 for performance
    #cov_mat = np.cov([row for row in matrix.T])
    for i in range(matrix.shape[0]):
        matrix[i, :] = matrix[i, :] - avgMatrix
    eigen_val,eigen_vec = np.linalg.eig(matrix.T.dot(matrix))
    eigen_pairs = [(eigen_val[i], eigen_vec[:, i], i) for i in range(len(eigen_val))]
    eigen_pairs.sort(key=lambda x:x[0], reverse=True)
    print eigen_vec.shape
    columns = []
    for i in range(num):
        columns.append(eigen_pairs[i][1].reshape(K, 1))
    matrix_w = np.hstack(tuple(columns)) # selected eigenvec
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


def knn(trainingFeatures, testFeatures, testLabels, trainingLabels, K = 5):
    predict = []
    for i in range(testFeatures.shape[0]):
        prediction = []
        j = 0
        while j < trainingFeatures.shape[0]:
            distance = np.linalg.norm(trainingFeatures[j, :]- testFeatures[i, :].T)
            heappush(prediction, (distance, trainingLabels[j]))
            j+=1
        predict.append(prediction)

    majority = []
    true_num = 0
    for i in range(len(predict)):
        dist = {}
        major_key = -1
        major = 0
        for j in range(K):
            key = heappop(predict[i])[1]
            if key in dist:
                dist[key]+=1
            else:
                dist.update({key: 1})
            if major < dist[key]:
                major_key = key
                major = dist[key]
        majority.append(major_key)

    for i in range(len(majority)):
        if majority[i] == testLabel[i]:
            true_num +=1

    print float(true_num) / float(testLabels.shape[0])


def off_the_shelf(trainingFeature, testingFeature, testingLabels, trainingLabels):
    p = PCA(n_components=200, whiten=True)
    X_train = p.fit(trainingFeature).transform(trainingFeature)
    X_test = p.transform(testingFeature)
    neighbor = KNeighborsClassifier(n_neighbors= 9).fit(X_train, trainingLabels)
    y_predict = neighbor.predict(X_test)
    print classification_report(testingLabels, y_predict)








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
    feature, label, test_size=0.5, random_state=42)

    num_of_train = trainingFeature.shape[0]
    print("Extracting the top %d eigenfaces from %d faces"
      % (num_of_train, 400))


    print "Test lable"

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


    #off-the-shelf implementation
    off_the_shelf(trainingFeature, testFeature, testLabel, trainingLabel)

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
    avgMatrix = np.mean(trainingFeature, 0)
    #trainingFeature = pca(trainingFeature, 4096, avgMatrix,1024)
    #testFeature = pca(testFeature, 4096, avgMatrix,1024)
    #knn(trainingFeature.T, testFeature.T, testLabel, trainingLabel)
