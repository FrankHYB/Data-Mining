import csv
import math
import operator
import numpy as np

#Set k = 4.
K = 4

#Define filenames for Iris output
IrisFile = 'Iris.csv'
IrisPreprocessed = 'Iris_Pre.csv'
iris_output_elud = 'Iris_output_elud.csv'
iris_output_cos = 'Iris_output_cos.csv'
header_iris = 'Transaction ID,1st,1-dist,2nd,2-dist,3rd,3-dist,4th,4-dist\n'

#Define filenames for Income output
incomeFile = 'income_tr.csv'
incomePreprocessed = 'Income_Pre.csv'
income_output_elud = 'Income_output_elud.csv'
income_output_cos = 'Income_output_cos.csv'
header_income = 'Transaction ID,1st,1-dist,2nd,2-dist,3rd,3-dist,4th,4-dist\n'


class IrisNode:

    #Initialize
    def __init__(self, data, i):
        self.sepal_length = float(data['sepal_length'][i])
        self.sepal_width = float(data['sepal_width'][i])
        self.petal_length = float(data['petal_length'][i])
        self.petal_width = float(data['petal_width'][i])


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


class OutputRow:
    def __init__(self, index, value):
        self.index = index
        self.value = value



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


#Preprocess income dataset.
#Translate strings to numbers; handle missing values; normalization
def preProcess_income(data):
    missingV = []

    ID = data['ID']
    age = data['age']

    #preprocessing workclass
    #Assign an unique number to each workclass
    workclass = data['workclass']
    cnt = 0
    wcDic = {}
    for i in range(len(workclass)):
        if '?' in workclass[i]:
            missingV.append(i)

        if(wcDic.has_key(workclass[i])):
            workclass[i] = wcDic.get(workclass[i])
        else:
            wcDic.update({workclass[i]:cnt})
            workclass[i] = cnt
            cnt = cnt+1

    fnlwgt = data['fnlwgt']

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


    edu_cat = data['education_cat']

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

    # preprocessing occupation
    #Assign an unique number to each occupation
    occupation = data['occupation']
    cnt = 0
    ocpDic = {}
    for i in range(len(occupation)):
        if '?' in occupation[i]:
            missingV.append(i)
        if(ocpDic.has_key(occupation[i])):
            occupation[i] = ocpDic.get(occupation[i])
        else:
            ocpDic.update({occupation[i]:cnt})
            occupation[i] = cnt
            cnt = cnt + 1

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


    capital_gain = data['capital_gain']
    capital_loss = data['capital_loss']
    hourPweek = data['hour_per_week']

    #preprocessing native_country
    #Assign an unique number to each country
    country = data['native_country']
    cnt = 0
    ctryDic = {}
    for i in range(len(country)):
        if '?' in country[i]:
            missingV.append(i)
        if (ctryDic.has_key(country[i])):
            country[i] = ctryDic.get(country[i])
        else:
            ctryDic.update({country[i]: cnt})
            country[i] = cnt
            cnt = cnt + 1

    #remove duplicate index in the missingV list.
    missingVIndex = set(missingV)
    #Update date with no missing value
    data = ignoreMissingValue(data, missingVIndex)

    #Normalize data
    data['ID'] = min_max_normalize(ID)
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




# input a list of header
#Write the preprocessed data into a file
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




#Write the result of Euclidean distance into a file
def write_to_file_eulid(header, processed_data, output_file):
    """
    :param header: a const str
    :param processed_data: list of nodes
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


#Write the result of cosine similarity into a file
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


def outlier_detection(column):
    """
    :param column: a column of a dataset
    :return: list of outliers' tranaction id
    """
    outlier = []
    u = np.mean(column)
    sd = np.std(column)
    for i in range(len(column)):
        if column[i] < u + 3*sd and column[i] > u - 3*sd:
            continue
        else:
            outlier.append(i)
    return outlier

#Ignore the rows which have a missinge value
def ignoreMissingValue(data,index):
    for i in range(len(data['ID'])):
        if i in index:
            del data['ID'][i]
            del data['age'][i]
            del data['workclass'][i]
            del data['fnlwgt'][i]
            del data['education'][i]
            del data['education_cat'][i]
            del data['marital_status'][i]
            del data['occupation'][i]
            del data['relationship'][i]
            del data['race'][i]
            del data['gender'][i]
            del data['capital_gain'][i]
            del data['capital_loss'][i]
            del data['hour_per_week'][i]
            del data['native_country'][i]
    return data


if __name__ == '__main__':

    #Iris data
    rawIris = readFile(IrisFile)
    data = preProcess_Iris(rawIris)
    write_preprocessed_data(data,IrisPreprocessed,['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    irises = []
    dist_matrix = []
    for i in range(len(data['sepal_length'])):
        irises.append(IrisNode(data, i))

    #Euclidean distance
    write_to_file_eulid(header_iris, irises, iris_output_elud)
    #Cosine similarity
    write_to_file_cos(header_iris, irises, iris_output_cos)


    #Income data
    rawIncome = readFile(incomeFile)

    #preprocessing
    incomeData = preProcess_income(rawIncome)
    incomeHeader = ['age', 'workclass', 'fnlwgt', 'education_cat','marital_status', 'occupation','relationship','race',
                    'gender','capital_gain','capital_loss','hour_per_week','native_country']
    write_preprocessed_data(incomeData,incomePreprocessed,incomeHeader)
    income = []
    dist_matrix_income = []
    for i in range(len(incomeData['ID'])):
        income.append(IncomeNode(incomeData,i))

    #Euclidean distance
    write_to_file_eulid(header_income,income,income_output_elud)
    #Cosine similarity
    write_to_file_cos(header_income,income,income_output_cos)
