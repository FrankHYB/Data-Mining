import csv
import math
import operator
import numpy as np
import matplotlib.pyplot as plt

K = 4
IrisFile = 'Iris.csv'
IrisPreprocessed = 'Iris_Pre.csv'
iris_output_elud = 'Iris_output_elud.csv'
iris_output_cos = 'Iris_output_cos.csv'
incomeFile = 'module2BusinessContext_v1.1.csv'
header_iris = 'Transaction ID,1st,1-dist,2nd,2-dist,3rd,3-dist,4th,4-dist\n'



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

    def cos_similarity(self, other):
        multi = self.sepal_length * other.sepal_length + self.sepal_width * other.sepal_width \
                 + self.petal_length * other.petal_length + self.petal_width * other.petal_width
        sq1 = self.sepal_length * self.sepal_length + self.sepal_width * self.sepal_width \
                + self.petal_length * self.petal_length + self.petal_width * self.petal_width
        sq2 = other.sepal_length * other.sepal_length + other.sepal_width * other.sepal_width \
                + other.petal_length * other.petal_length + other.petal_width * other.petal_width
        return multi / (math.sqrt(sq1) * math.sqrt(sq2))


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
    sepal_length = data['sepal_length']
    sepal_width =  data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']

    data['sepal_length'] = min_max_normalize(sepal_length)
    data['sepal_width'] = min_max_normalize(sepal_width)
    data['petal_length'] = min_max_normalize(petal_length)
    data['petal_width'] = min_max_normalize(petal_width)

    return data

def min_max_normalize(data):
    data = map(float,data)
    minV = min(data)
    maxV = max(data)

    for i in range(len(data)):
        data[i] = ((data[i] - minV)/(maxV - minV))

    return data



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

# input a list of header
#
def write_preprocessed_data(data, filename, header):
    with open(filename, 'w') as f:
        f.write(''.join(str(h) for h in header) + '\n')
        length = len(data[header[0]])
        for i in range(length):
            for j in range(len(header)):
                if j!=0:
                    f.write(',' + str(data[header[j]][i]))
                else:
                    f.write(str(data[header[j]][i]))
            f.write('\n')





def write_to_file_eulid(header, processed_data, output_file):
    """
    :param header: a const str
    :param processed_data: list of iris node
    :param output_file: a str
    :return: none
    """
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

def write_to_file_cos(header, processed_data, output_file):
    with open(output_file, 'w') as f:
        f.write(header)
        for i in range(len(processed_data)):
            dist_matrix = []
            for j in range(len(processed_data)):
                if j != i:
                    dist_matrix.append(OutputRow(j, processed_data[i].cos_similarity(processed_data[j])))
            dist_matrix.sort(key = operator.attrgetter('value'), reverse=True)
            for k in range(K):
                if k == 0:
                    f.write(str(i) + ',' + str(dist_matrix[k].index) + ',' + str(dist_matrix[k].value))
                else:
                    f.write(',' + str(dist_matrix[k].index) + ',' + str(dist_matrix[k].value))
            f.write('\n')



if __name__ == '__main__':

    #rawIncome = readFile(incomeFile)
    rawIris = readFile(IrisFile)
    #preProcess_income(rawIncome)
    data = preProcess_Iris(rawIris)
    write_preprocessed_data(data,IrisPreprocessed,['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    irises = []
    dist_matrix = []
    for i in range(len(data['sepal_length'])):
        irises.append(IrisNode(data, i))

    write_to_file_eulid(header_iris, irises, iris_output_elud)
    write_to_file_cos(header_iris, irises, iris_output_cos)


