import csv
import math
import operator
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
#Set k = 5.
K = 8

#Define filenames for Iris output
IrisFile = 'Iris.csv'
IrisTest = 'Iris_Test.csv'

#Define filenames for Income output
incomeFile = 'income_tr.csv'
incomeTest = 'income_te.csv'

Header = 'TransactionID, Actual Class, Predicted Class, Posterior Probability\n'
Iris_Prediction = 'Iris_Prediction.csv'
Income_Prediction = 'Income_Prediction.csv'

Iris_Unweighted_Prediction = 'Iris_Unweighted_Prediction'
Income_Unweighted_Prediction = 'Income_Unweighted_Prediction'

Iris_Weighted_Prediction = 'Iris_Weighted_Prediction.csv'
Income_Weighted_Prediction = 'Income_Weighted_Prediction.csv'

class IncomeNode:

    #Initialize
    def __init__(self, data, i):
        self.ID = data['ID'][i]
        self.age = data['age'][i]
        self.workclass = data['workclass'][i]
        self.fnlwgt = data['fnlwgt'][i]
        self.edu = data['education'][i]
        self.edu_cat = data['education_cat'][i]
        self.marital = data['marital_status'][i]
        self.occupation = data['occupation'][i]
        self.relationship = data['relationship'][i]
        self.race = data['race'][i]
        self.gender = data['gender'][i]
        self.capital_gain = data['capital_gain'][i]
        self.capital_loss = data['capital_loss'][i]
        self.hourPweek = data['hour_per_week'][i]
        self.country = data['native_country'][i]
        self.clas = data['class'][i]



    #Euclidean distance
    def eulid(self,other):
        x = [self.age,self.workclass,self.fnlwgt,self.edu_cat,self.marital,self.occupation,self.relationship,self.race,
             self.gender,self.capital_gain,self.capital_loss,self.hourPweek,self.country]
        y = [other.age,other.workclass,other.fnlwgt,other.edu_cat,other.marital,other.occupation,other.relationship,other.race,
             other.gender,other.capital_gain,other.capital_loss,other.hourPweek,other.country]
        total = sum(math.pow(element1 - element2, 2) for element1, element2 in zip(x, y))
        dist = math.sqrt(total)
        return round(dist,5)

    #Cosine similarity
    def cos_similarity(self, other):
        x = [self.age, self.workclass, self.fnlwgt, self.edu_cat, self.marital, self.occupation, self.relationship,self.race,
             self.gender, self.capital_gain, self.capital_loss, self.hourPweek, self.country]

        y = [other.age, other.workclass, other.fnlwgt, other.edu_cat, other.marital, other.occupation, other.relationship, other.race,
         other.gender, other.capital_gain, other.capital_loss, other.hourPweek, other.country]
        numerator = sum(element1 * element2 for element1, element2 in zip(x, y))
        magX = math.sqrt(sum([element1 * element1 for element1 in x]))
        magY = math.sqrt(sum([element2 * element2 for element2 in y]))
        return round(numerator / float(magX*magY), 5)


    def set_prediction(self, prediction, probability):
        self.prediction = prediction
        self.posterior = probability

class IrisNode:

    #Initialize
    def __init__(self, data, i):
        self.sepal_length = float(data['sepal_length'][i])
        self.sepal_width = float(data['sepal_width'][i])
        self.petal_length = float(data['petal_length'][i])
        self.petal_width = float(data['petal_width'][i])
        self.clas = data['class'][i]



    #Euclidean distance
    def eulid(self, other):
        sum = 0
        sum = (self.sepal_length - other.sepal_length) * (self.sepal_length - other.sepal_length)
        sum += (self.sepal_width - other.sepal_width) * (self.sepal_width - other.sepal_width)
        sum += (self.petal_length - other.petal_length) * (self.petal_length - other.petal_length)
        sum += (self.petal_width - other.petal_width) * (self.petal_width - other.petal_width)
        return math.sqrt(sum)


    #Cosine similarity
    def cos_similarity(self, other):
        multi = self.sepal_length * other.sepal_length + self.sepal_width * other.sepal_width \
                 + self.petal_length * other.petal_length + self.petal_width * other.petal_width
        sq1 = self.sepal_length * self.sepal_length + self.sepal_width * self.sepal_width \
                + self.petal_length * self.petal_length + self.petal_width * self.petal_width
        sq2 = other.sepal_length * other.sepal_length + other.sepal_width * other.sepal_width \
                + other.petal_length * other.petal_length + other.petal_width * other.petal_width
        return multi / (math.sqrt(sq1) * math.sqrt(sq2))


    def set_prediction(self, prediction, probability):
        self.prediction = prediction
        self.posterior = probability



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

#Preprocess Iris dataset.
#Normalization
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


def mostFrequent(data):
    return max(set(data), key=data.count)

#Preprocess income dataset.
#Translate strings to numbers; handle missing values; normalization; outlier
def preProcess_income(data, ifTestData):

    ID = data['ID']
    ID = map(float,ID)
    age = data['age']
    age = map(float,age)


    #preprocessing workclass
    #Assign an unique number to each workclass
    workclass = data['workclass']
    mostCom = mostFrequent(workclass)

    cnt = 0
    wcDic = {}
    for i in range(len(workclass)):
        if '?' in workclass[i]:
            workclass[i] = mostCom

        if(wcDic.has_key(workclass[i])):
            workclass[i] = wcDic.get(workclass[i])
        else:
            wcDic.update({workclass[i]:cnt})
            workclass[i] = cnt
            cnt = cnt+1
    workclass = map(float,workclass)




    fnlwgt = data['fnlwgt']
    fnlwgt = map(float,fnlwgt)


    #preprocessing education
    #Assign an unique number to each education
    edu = data['education']
    cnt = 0
    edDic = {}
    for i in range(len(edu)):
        if(edDic.has_key(edu[i])):
            edu[i] = edDic.get(edu[i])
        else:
            edDic.update({edu[i]:cnt})
            edu[i] = cnt
            cnt = cnt + 1
    edu = map(float,edu)


    edu_cat = data['education_cat']
    edu_cat = map(float,edu_cat)


    #preprocessing marital_Status
    #Assign an unique number to each marital_status
    marital = data['marital_status']
    cnt = 0
    marDic = {}
    for i in range(len(marital)):
        if (marDic.has_key(marital[i])):
            marital[i] = marDic.get(marital[i])
        else:
            marDic.update({marital[i]: cnt})
            marital[i] = cnt
            cnt = cnt + 1
    marital = map(float, marital)

    # preprocessing occupation
    #Assign an unique number to each occupation
    occupation = data['occupation']
    mostCom = mostFrequent(occupation)
    cnt = 0
    ocpDic = {}
    for i in range(len(occupation)):
        if '?' in occupation[i]:
            occupation[i] = mostCom
        if(ocpDic.has_key(occupation[i])):
            occupation[i] = ocpDic.get(occupation[i])
        else:
            ocpDic.update({occupation[i]:cnt})
            occupation[i] = cnt
            cnt = cnt + 1
    occupation = map(float, occupation)

    # preprocessing relationship
    #Assign an unique number to each relationship
    relationship = data['relationship']
    cnt = 0
    reDic = {}
    for i in range(len(relationship)):
        if (reDic.has_key(relationship[i])):
            relationship[i] = reDic.get(relationship[i])
        else:
            reDic.update({relationship[i]: cnt})
            relationship[i] = cnt
            cnt = cnt + 1
    relationship = map(float, relationship)

    # preprocessing race
    #Assign an unique number to each race
    race = data['race']
    cnt = 0
    raceDic = {}
    for i in range(len(race)):
        if (raceDic.has_key(race[i])):
            race[i] = raceDic.get(race[i])
        else:
            raceDic.update({race[i]: cnt})
            race[i] = cnt
            cnt = cnt + 1
    race = map(float, race)

    # preprocessing gender
    #Assign an unique number to each gender
    gender = data['gender']
    cnt = 0
    genDic = {}
    for i in range(len(gender)):
        if (genDic.has_key(gender[i])):
            gender[i] = genDic.get(gender[i])
        else:
            genDic.update({gender[i]: cnt})
            gender[i] = cnt
            cnt = cnt + 1
    gender = map(float, gender)

    capital_gain = data['capital_gain']
    capital_gain = map(float, capital_gain)
    capital_loss = data['capital_loss']
    capital_loss = map(float,capital_loss)
    hourPweek = data['hour_per_week']
    hourPweek = map(float,hourPweek)

    #preprocessing native_country
    #Assign an unique number to each country
    country = data['native_country']
    mostCom = mostFrequent(country)

    cnt = 0
    ctryDic = {}
    for i in range(len(country)):
        if '?' in country[i]:
            country[i] = mostCom

        if (ctryDic.has_key(country[i])):
            country[i] = ctryDic.get(country[i])
        else:
            ctryDic.update({country[i]: cnt})
            country[i] = cnt
            cnt = cnt + 1
    country = map(float,country)

    if not ifTestData:
        #get index of outliers
        age = outlier_detection(age)
        workclass = outlier_detection(workclass)
        fnlwgt = outlier_detection(fnlwgt)
        edu = outlier_detection(edu)
        edu_cat = outlier_detection(edu_cat)
        marital = outlier_detection(marital)
        occupation = outlier_detection(occupation)
        relationship = outlier_detection(relationship)
        race = outlier_detection(race)
        gender = outlier_detection(gender)
        hourPweek = outlier_detection(hourPweek)
        country = outlier_detection(country)


    #Normalize data
    data['age'] = min_max_normalize(age)
    data['workclass'] = min_max_normalize(workclass)
    data['fnlwgt'] = min_max_normalize(fnlwgt)
    data['education'] = min_max_normalize(edu)
    data['education_cat'] = min_max_normalize(edu_cat)
    data['marital_status'] = min_max_normalize(marital)
    data['occupation'] = min_max_normalize(occupation)
    data['relationship'] = min_max_normalize(relationship)
    data['race'] = min_max_normalize(race)
    data['gender'] = min_max_normalize(gender)
    data['capital_gain'] = min_max_normalize(capital_gain)
    data['capital_loss'] = min_max_normalize(capital_loss)
    data['hour_per_week'] = min_max_normalize(hourPweek)
    data['native_country'] = min_max_normalize(country)

    return data


#Min_max normalization
def min_max_normalize(data):
    data = map(float,data)
    minV = min(data)
    maxV = max(data)

    for i in range(len(data)):
        data[i] = ((data[i] - minV)/(maxV - minV))

    return data


#Detect outlier and assign the most frequent value to outlier
def outlier_detection(column):
    """
    :param column: a column of a dataset
    :return: list of outliers' tranaction id
    """
    u = np.mean(column)
    sd = np.std(column)
    mostCom = mostFrequent(column)

    for i in range(len(column)):
        if column[i] < u + 3*sd and column[i] > u - 3*sd:
            continue
        else:
            column[i] = mostCom
    return column



#Find the K-nearest neighbors
def findNeighbors(testData, data, cos = False):
    neighbors = []
    dist = []

    #Calculate Euclidean Distance between one test data and all training data
    for i in range(len(data)):
        if cos:
            dist.append((testData.cos_similarity(data[i]),data[i]))
        else:
            dist.append((testData.eulid(data[i]),data[i]))

    if cos:
        dist.sort(key=operator.itemgetter(0), reverse= True)
    else:
        dist.sort(key=operator.itemgetter(0))


    #Get first k neighbors
    for k in range(K):
        neighbors.append(dist[k])

    return neighbors # tuple[0] distance tuple[1] neighor node


#KNN and calculate posterior probability
def posterior_knn(neighbor, weight = False, cos = False):
    #Calculate the number of classifiers within K neighbors
    classifer = {}
    weighted_k = K
    if weight:
        weighted_k = 0
    for i in range(len(neighbor)):

        className = neighbor[i][1].clas
        if className in classifer:
            if weight:
                if cos:
                    classifer[className] += neighbor[i][0]
                    weighted_k += neighbor[i][0]
                else:
                    classifer[className] += 1 / neighbor[i][0]
                    weighted_k += 1 / neighbor[i][0]

            else:
                classifer[className] += 1

        else:
            if weight:
                if cos:
                    classifer[className] = neighbor[i][0]
                    weighted_k += neighbor[i][0]
                else:
                    classifer[className] = 1 / neighbor[i][0]
                    weighted_k += 1 / neighbor[i][0]
            else:
                classifer[className] = 1



    prediction = ''
    probability = 0.0


    for key in classifer:
        if ((classifer[key] * 1.0 / weighted_k) > probability):
            probability = classifer[key] * 1.0 / weighted_k
            prediction = key



    return prediction,probability


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
    print 'Use for confusion matrices'
    print Counter(table)
    accuracy = float(correct)/len(nodes)
    print "Accuracy for " + filename + ' ' + str(accuracy) + '\n'

#off-the-shelf implementation - sklearn
def sk_implementation(train_nodes, test_nodes, income_data = False):
    data_X = []
    cla_Y = []
    for node in train_nodes:
        one_node = []
        for key,value in node.__dict__.items():
            if key == 'clas' or key == 'prediction' or key == 'posterior':
                continue
            one_node.append(float(value))
        data_X.append(one_node)
        #transform class labels to numbers
        if income_data:
            if node.clas == ' <=50K':
                cla_Y.append(0)
            else:
                cla_Y.append(1)
        else:
            if node.clas == 'Iris-setosa':
                cla_Y.append(0)
            elif node.clas == 'Iris-versicolor':
                cla_Y.append(1)
            else:
                cla_Y.append(2)

    test_X = []
    cla_test = []
    for node in test_nodes:
        one_node = []
        for key,value in node.__dict__.items():
            if key == 'clas' or key == 'prediction' or key == 'posterior':
                continue
            one_node.append(float(value))
        test_X.append(one_node)
        #transform class to label to numbers
        if income_data:
            if node.clas == ' <=50K':
                cla_test.append(0)
            else:
                cla_test.append(1)
        else:
            if node.clas == 'Iris-setosa':
                cla_test.append(0)
            elif node.clas == 'Iris-versicolor':
                cla_test.append(1)
            else:
                cla_test.append(2)
    print cla_test
    neigh = KNeighborsClassifier(n_neighbors= K, algorithm = 'auto', p = 2)
    neigh.fit(data_X, cla_Y)
    correct = 0
    for i in range(len(test_X)):
        label = neigh.predict(test_X[i])[0]
        if label == cla_test[i]:
            correct+=1
    if income_data:
        print 'Off-shelf-Implementation Accuracy for Income Data:' + str(correct * 1.0/len(test_X))
    else:
        print 'Off-shelf-Implementation Accuracy for Iris Data:' + str(correct * 1.0/len(test_X))


if __name__ == '__main__':

    #Load Iris data
    rawIris = readFile(IrisFile)
    testIris = readFile(IrisTest)

    #Pre-processing iris
    IrisData = preProcess_Iris(rawIris)
    testIrisData = preProcess_Iris(testIris)

    #Initialize
    irises = []
    irisesTest = []

    for i in range(len(IrisData['sepal_length'])):
        irises.append(IrisNode(IrisData, i))

    for i in range(len(testIrisData['sepal_length'])):
        irisesTest.append(IrisNode(testIrisData, i))

    #Eliud Unweighted_KNN
    for i in range(len(irisesTest)):
        neighbor = findNeighbors(irisesTest[i],irises)
        pred,prob = posterior_knn(neighbor)
        irisesTest[i].set_prediction(pred, prob)

    write_to_file(Iris_Prediction,Header,irisesTest)
    #Cos unweighted KNN
    for i in range(len(irisesTest)):
        neighbor = findNeighbors(irisesTest[i],irises, True)
        pred,prob = posterior_knn(neighbor,False,True)
        irisesTest[i].set_prediction(pred, prob)

    write_to_file(Iris_Unweighted_Prediction,Header,irisesTest)

    #Weighted KNN

    for i in range(len(irisesTest)):
        neighbor = findNeighbors(irisesTest[i],irises, True)
        pred,prob = posterior_knn(neighbor, True ,True)
        irisesTest[i].set_prediction(pred, prob)

    write_to_file(Iris_Weighted_Prediction,Header,irisesTest)

    sk_implementation(irises, irisesTest)

    #Income data
    rawIncome = readFile(incomeFile)
    testRawIncome = readFile(incomeTest)

    #preprocessing
    ifTestData = 0
    incomeData = preProcess_income(rawIncome,ifTestData)
    ifTestData = 1
    testIncomeData = preProcess_income(testRawIncome,ifTestData)

    #Initialize
    income = []
    testIncome = []
    for i in range(len(incomeData['ID'])):
        income.append(IncomeNode(incomeData, i))
    for i in range(len(testIncomeData['ID'])):
        testIncome.append(IncomeNode(testIncomeData,i))

    # KNN
    for i in range(len(testIncome)):
        neighbor = findNeighbors(testIncome[i], income)
        pred, prob = posterior_knn(neighbor)
        testIncome[i].set_prediction(pred,prob)

    write_to_file(Income_Prediction, Header, testIncome)

    #Cos unweighted KNN
    for i in range(len(testIncome)):
        neighbor = findNeighbors(testIncome[i],income, True)
        pred,prob = posterior_knn(neighbor,False,True)
        testIncome[i].set_prediction(pred, prob)

    write_to_file(Income_Unweighted_Prediction,Header,testIncome)

    #Cos Weighted KNN
    for i in range(len(testIncome)):
        neighbor = findNeighbors(testIncome[i], income, True)
        pred, prob = posterior_knn(neighbor, True,True)
        testIncome[i].set_prediction(pred,prob)

    write_to_file(Income_Weighted_Prediction, Header, testIncome)

    sk_implementation(income, testIncome, True)


    # 1: <=50K, 0: >50K get roc plot
    label = []
    score = []
    for node in testIncome:
        if node.clas == ' <=50K':
            label.append(0)
            score.append(node.posterior)
        else:
            label.append(1)
            score.append(node.posterior)

    tpr, fpr, threshold = roc_curve(label, score, pos_label = 1)
    roc_auc = auc(fpr, tpr)
    plt.title('Income roc curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr, tpr, label = 'ROC curve (area = 0.2f)' %roc_auc)
    plt.show()

