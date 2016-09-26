import math
import operator
import numpy as np
import csv
import sys
K = 5
classification_Iris = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
Iris_Header = 'TransactionID, Actual Class, Predicted Class, Posterior Probability\n'
Iris_Prediction = 'Iris_Prediction.csv'

class Dist_Classifier:
    #initialize
    def __init__(self, record):
        self.id = record['Transaction ID']
        self.neighbor = [] # string type, k cloest neighbors' id
        for i in range(K):
            if i == 0:
                self.neighbor.append(record[str((i+1))+'st']) #start from 1
            elif i == 1:
                self.neighbor.append(record[str((i+1))+'nd']) #start from 1
            elif i == 2:
                self.neighbor.append(record[str((i+1))+'rd']) #start from 1
            else:
                self.neighbor.append(record[str((i+1))+'th']) #start from 1

        self.cla = record['class']
        self.prediction = ""
        self.posterior = 0.0

    def set_prediction(self, prediction, probability):
        self.prediction = prediction
        self.posterior = probability
    def get_neighbors(self):
        return self.neighbor

#Read data files
def readFile(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        dist = []
        for row in reader:
            dist.append(row)
    return dist


def posterior_knn(classifer, node, all_nodes):
    num = {}
    for ele in classifer:
        num[ele] = 0

    for ele in node.get_neighbors():
        num[all_nodes[int(ele)].cla] +=1
        #print ele + '\n'
    prediction = ''
    probability =0.0

    for key in num:
        if((num[key] * 1.0 / K) > probability):
            probability = num[key] * 1.0 / K
            prediction = key

    node.set_prediction(prediction, probability)

def write_to_file(filename, header, nodes):
    with open(filename, 'w') as f:
        f.write(header)
        for node in nodes:
            f.write(node.id + ',' + node.cla + ',' + node.prediction + ',' + str(node.posterior))
            f.write('\n')

if __name__ == '__main__':
    #if len(sys.argv) == 0:
    #    print 'Please input the proximity file'
    #else:
        nodes = []
        proximities = readFile('Iris_output_elud.csv') #a list of rows
        for dict in proximities:
            nodes.append(Dist_Classifier(dict)) #initialize the node by row
        for node in nodes:
            posterior_knn(classification_Iris, node, nodes)
        write_to_file(Iris_Prediction, Iris_Header, nodes)


