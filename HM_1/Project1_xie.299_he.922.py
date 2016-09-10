import csv
import math
import operator
import numpy as np
import matplotlib.pyplot as plt

K = 4
IrisFile = 'Iris.csv'
iris_output = 'Iris_output.csv'
incomeFile = 'module2BusinessContext_v1.1.csv'
header_iris = 'Transaction ID,1st,1-dist,2nd,2-dist,3rd,3-dist,4th,4-dist\n'
# Normalize the columns in order to minimize the difference
max_sepal_length = 0.0
min_sepal_length = 0.0
max_sepal_width = 0.0
min_sepal_width = 0.0
max_petal_length = 0.0
min_petal_length = 0.0
max_petal_width = 0.0
min_petal_width = 0.0

diff_sepal_length, diff_sepal_width = 0.0, 0.0
diff_petal_length, diff_petal_width = 0.0, 0.0

class IrisNode:

    def __init__(self, data, i):
        self.sepal_length = float(data['sepal_length'][i])
        self.sepal_width = float(data['sepal_width'][i])
        self.petal_length = float(data['petal_length'][i])
        self.petal_width = float(data['petal_width'][i])

    def eulid(self, other):
        sum = 0
        sum = (self.sepal_length - other.sepal_length) * (self.sepal_length - other.sepal_length)
        sum += (self.sepal_width - other.sepal_width) * (self.sepal_width - other.sepal_width)
        sum += (self.petal_length - other.petal_length) * (self.petal_length - other.petal_length)
        sum += (self.petal_width - other.petal_width) * (self.petal_width - other.petal_width)
        return math.sqrt(sum)

    #TODO: Another proximity

class OutputRow:
    def __init__(self, index, value):
        self.index = index
        self.value = value


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


def preProcess_Iris(data):
    #columns
    sepal_length = [float(ele) for ele in data['sepal_length']]
    sepal_width = [float(ele) for ele in data['sepal_width']]
    petal_length = [float(ele) for ele in data['petal_length']]
    petal_width = [float(ele) for ele in data['petal_width']]

    sepal_length.sort()
    max_sepal_length = sepal_length[len(sepal_length) - 1]
    min_sepal_length = sepal_length[0]
    diff_sepal_length = max_sepal_length - min_sepal_length

    sepal_width.sort()
    max_sepal_width = sepal_width[len(sepal_width) - 1]
    min_sepal_width = sepal_width[0]
    diff_sepal_width = max_sepal_width - min_sepal_width

    petal_length.sort()
    max_petal_length = petal_length[len(petal_length) - 1]
    min_petal_length = petal_length[0]
    diff_petal_length = max_petal_length - min_petal_length

    petal_width.sort()
    max_petal_width = petal_width[len(petal_width) - 1]
    min_petal_width = petal_width[0]
    diff_petal_width = max_petal_width - min_petal_width

    data['sepal_length'] = min_max_normalization(sepal_length, min_sepal_length, max_sepal_length)
    data['sepal_width'] = min_max_normalization(sepal_width, min_sepal_width, max_sepal_width)
    data['petal_length'] = min_max_normalization(petal_length, min_petal_length, max_petal_length)
    data['petal_width'] = min_max_normalization(petal_width, min_petal_width, max_petal_width)

    return data

def min_max_normalization(column, min, max):
    for i in range(len(column)):
        column[i] = (column[i] - min) / (max - min)
    return column



def preProcess_income(data):
    ID = data['ID']
    age = data['age']
    workclass = data['workclass']
    fnlwgt = data['fnlwgt']
    edu = data['education']
    edu_cat = data['education_cat']
    marital = data['marital_status']
    occupation = data['occupation']
    relationship = data['relationship']
    race = data['race']
    gender = data['gender']
    capital_gain = data['capital_gain']
    capital_loss = data['capital_loss']
    hourPweek = data['hour_per_week']
    country = data['native_country']

def write_to_file_eulid(header, processed_data, output_file):
    with open(output_file, 'w') as f:
        f.write(header)
        for i in range(len(processed_data)):
            dist_matrix = []

            for j in range(len(processed_data)):
                if j!=i:
                    dist_matrix.append(OutputRow(j, processed_data[i].eulid(processed_data[j])))

            dist_matrix.sort(key=operator.attrgetter('value'))
            for k in range(K):
                if k == 0:
                    f.write(str(i) + ',' + str(dist_matrix[k].index) +',' +str(dist_matrix[k].value))
                else:
                    f.write(',' + str(dist_matrix[k].index) +',' +str(dist_matrix[k].value))
            f.write('\n')


if __name__ == '__main__':

    #rawIncome = readFile(incomeFile)
    rawIris = readFile(IrisFile)
    #preProcess_income(rawIncome)
    data = preProcess_Iris(rawIris)
    irises = []
    dist_matrix = []
    for i in range(len(data['sepal_length'])):
        irises.append(IrisNode(data, i))

    write_to_file_eulid(header_iris, irises, iris_output)



