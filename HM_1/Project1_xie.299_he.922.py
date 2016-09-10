import csv
import math
import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self, data):
        self.sepal_length = float(data[0])
        self.sepal_width = float(data[1])
        self.petal_length = float(data[2])
        self.petal_width = float(data[3])

    def eulid(self, other):
        sum = 0
        sum = (self.sepal_length - other.sepal_length) * (self.sepal_length - other.sepal_length)
        sum += (self.sepal_width - other.sepal_width) * (self.sepal_width - other.sepal_width)
        sum += (self.petal_length - other.petal_length) * (self.petal_length - other.petal_length)
        sum += (self.petal_width - other.petal_width) * (self.petal_width - other.petal_width)
        return math.sqrt(sum)



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
    sepal_width = data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']

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



#Main
IrisFile = 'Iris.csv'
incomeFile = 'module2BusinessContext_v1.1.csv'
#rawIncome = readFile(incomeFile)
rawIris = readFile(IrisFile)
#preProcess_income(rawIncome)
#preProcess_Iris(rawIris)

