import csv
import numpy as np
import matplotlib.pyplot as plt

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
    sepal_length = data['sepal_length']
    sepal_width = data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']



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
rawIncome = readFile(incomeFile)
rawIris = readFile(IrisFile)
preProcess_income(rawIncome)
preProcess_Iris(rawIris)

