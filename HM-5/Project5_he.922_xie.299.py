import csv
import sys
import math
import random
import operator
import numpy as np

num_clusters_easy = 2;
num_clusters_hard = 4;
initial_centroid_easy = []
initial_centroid_hard = []
initial_centroid_wine = []
two_dim_easy = 'TwoDimEasy.csv'
two_dim_hard = 'TwoDimHard.csv'
wine = 'wine.csv'

class TwoDimNode:

    def __init__(self, data, i):
        self.id = data['ID'][i]
        self.x1 = data['X.1'][i]
        self.x2 = data['X.2'][i]
        self.actual_cluster = data['cluster'][i]
        self.prob = 0
        self.predict = 0
        self.centroid = None

    def eulid(self, other):
        x = [self.x1, self.x2]
        y = [other.x1, other.x2]
        total = sum(math.pow(element1-element2, 2) for element1, element2 in zip(x,y))
        res = math.sqrt(total)
        return round(res, 5)



class WineNode:
    def __init__(self,data, i):
        self.id = data['ID'][i]
        self.fx_acidity = data['fx_acidity'][i]
        self.vol_acidity = data['vol_acidity'][i]
        self.citric_acid = data['citric_acid'][i]
        self.resid_sugar = data['resid_sugar'][i]
        self.chlorides = data['chlorides'][i]
        self.free_sulf_d = data['free_sulf_d'][i]
        self.tot_sulf_d = data['tot_sulf_d'][i]
        self.density = data['density'][i]
        self.pH = data['pH'][i]
        self.sulph = data['sulph'][i]
        self.alcohol = data['alcohol'][i]
        self.actual_cluster = data['class'][i]
        self.prob = 0
        self.predict = ""
        self.centroid = None
    def eulid(self, other):
        x = [self.fx_acidity, self.vol_acidity, self.citric_acid, self.resid_sugar, self.chlorides, self.free_sulf_d /
             self.tot_sulf_d, self.density, self.pH, self.sulph, self.alcohol]
        y = [other.fx_acidity, other.vol_acidity, other.citric_acid, other.resid_sugar, other.chlorides, other.free_sulf_d /
             other.tot_sulf_d, other.density, other.pH, other.sulph, other.alcohol]
        total = sum(math.pow(element1-element2, 2) for element1, element2 in zip(x,y))
        res = math.sqrt(total)
        return round(res, 5)



#Min_max normalization
def min_max_normalize(data):
    data = map(float,data)
    minV = min(data)
    maxV = max(data)

    for i in range(len(data)):
        data[i] = ((data[i] - minV)/(maxV - minV))

    return data


#Generate output file
def write_to_file(filename, header,  nodes):
    accuracy = 0.0
    correct = 0
    table = []
    with open(filename, 'w') as f:
        f.write(header)
        count = 1
        for node in nodes:
            f.write(str(count) + ',' + node.clas + ',' + node.prediction + ',' + str(node.posterior))
            f.write('\n')
            count += 1
            table.append((node.clas,node.prediction))
            if node.clas == node.prediction:
                correct += 1

    accuracy = float(correct)/len(nodes)
    print "Accuracy for " + filename + ' ' + str(accuracy) + '\n'

#Read data files
def readFile(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        data = {}
        for row in reader:
            for header, value in row.items():
                try:
                    data[header].append(value)
                except KeyError:
                    data[header] = [value]

    return data

#convert a list number to dictionary in order to get new node
def construct_centroid(points):
    new_dict = {'ID':0}
    if len(points) == 2:
        new_dict.update({'X.1':[points[0]]})
        new_dict.update({'X.2':[points[1]]})
        new_dict.update({'cluster':[1]})
        centroid = TwoDimNode(new_dict,0)
    else:
        new_dict.update({'fx_acidity':[points[0]]})
        new_dict.update({'vol_acidity':[points[1]]})
        new_dict.update({'citric_acid':[points[2]]})
        new_dict.update({'resid_sugar':[points[3]]})
        new_dict.update({'chlorides':[points[4]]})
        new_dict.update({'free_sulf_d':[points[5]]})
        new_dict.update({'tot_sulf_d':[points[6]]})
        new_dict.update({'density':[points[7]]})
        new_dict.update({'pH':[points[8]]})
        new_dict.update({'sulph':[points[9]]})
        new_dict.update({'alcohol':[points[10]]})
        new_dict.update({'class':[1]})
        centroid = WineNode(new_dict,0)
    return centroid



def initial_centroid(data, opt, k, nodes):
    listOfCentroid = [];

    if opt == 1:
        length = len(data.values()[0])
        for i in range(k):
            radIndex = random.randint(0, length-1)
            #If a duplicate index is generated, random another index
            while nodes[radIndex] in listOfCentroid:
                radIndex = random.randint(0, length - 1)
            #return a list of nodes
            listOfCentroid.append(nodes[radIndex])

    elif opt == 2:
        #Find k centroids
        for i in range(k):
            points = []
            #Random a number for each attribute
            for key, value in data.items():
                if key != 'ID' and key != 'class' and key !='cluster':
                    #find the max and min values in a column.
                    minV = min(value)
                    maxV = max(value)
                    rad = random.uniform(minV,maxV)
                    points.append(rad)


            #return a list of points(num)
            listOfCentroid.append(points)



    elif opt == 3:
        centroid = nodes[random.randint(0, len(nodes) - 1)]
        centroid.prob = 0
        listOfCentroid.append(centroid)
        for i in range(k - 1):
            for ele in nodes:
                if ele in listOfCentroid:
                    continue
                dist = sys.maxint
                for c in listOfCentroid:
                    temp = ele.eulid(c)
                    if temp < dist:
                        dist = temp
                ele.prob = math.pow(dist, 2)
            weight = [element.prob for element in nodes]
            arr = np.random.choice(len(nodes), 1, weight)
            #Return a list of nodes
            listOfCentroid.append(nodes[arr[0]])




    #else:
    return listOfCentroid


def compute_sse(nodes, dict_centroid):
    num_of_cluster = len(dict_centroid)
    sse = []
    for i in range(num_of_cluster):
        every_sse = 0
        centroid = dict_centroid[i+1] # map cluster number -> centroid
        for node in nodes:
            if node.actual_cluster != i+1:
                continue
            every_sse += math.pow(node.eulid(centroid), 2)
        sse.append(every_sse)
    return sum(sse), sse


if __name__ == '__main__':
    K = 2
    if len(sys.argv) == 2:
        K = int(sys.argv[1])
    data_easy = readFile(two_dim_easy)
    data_hard = readFile(two_dim_hard)
    data_wine = readFile(wine)
    data_wine.pop('quality')
    for key, value in data_easy.items():
        if key == 'cluster' or key == 'ID':
            continue
        data_easy[key] = min_max_normalize(value)

    for key, value in data_hard.items():
        if key == 'cluster'or key == 'ID':
            continue
        data_hard[key] = min_max_normalize(value)

    for key, value in data_wine.items():
        if key == 'ID':
            continue
        if key == 'class':
            new_c = []
            # transform to integer value
            for ele in value:
                if ele == 'Low':
                    new_c.append(1)
                else:
                    new_c.append(2)
        else:
            data_wine[key] = min_max_normalize(value)

    easy_nodes = []
    hard_nodes = []
    wine_nodes = []
    for i in range(len(data_easy['ID'])):
        easy_nodes.append(TwoDimNode(data_easy, i))

    for i in range(len(data_hard['ID'])):
        hard_nodes.append(TwoDimNode(data_hard, i))

    for i in range(len(data_wine['ID'])):
        wine_nodes.append(WineNode(data_wine, i))
