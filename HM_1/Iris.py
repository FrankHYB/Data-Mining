import math
iris_filename = "Iris.csv"
output_filename = "Iris_result.csv"


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


class Node:

    def __init__(self, data):
        self.sepal_length = float(data[0])
        self.sepal_width = float(data[1])
        self.petal_length = float(data[2])
        self.petal_width = float(data[3])



    def eulid(self, other_iris):
        sum = 0
        sum = (self.sepal_length - other_iris.sepal_length) * (self.sepal_length - other_iris.sepal_length)
        sum += (self.sepal_width - other_iris.sepal_width) * (self.sepal_width - other_iris.sepal_width)
        sum += (self.petal_length - other_iris.petal_length) * (self.petal_length - other_iris.petal_length)
        sum += (self.petal_width - other_iris.petal_width) * (self.petal_width - other_iris.petal_width)
        return math.sqrt(sum)


        #TODO: other way to calculate the distance matrix
        #def other(self):




if __name__ == '__main__':
    with open (iris_filename) as f:
        content = f.readlines()
        length = len(content)
        column = []
        for j in range(4):
            for i in range(length):
                column[i] = content[i].split(',')[j]

            column.sort()
            if j == 0:
                max_sepal_length = column[length-1]
                min_sepal_length = column[0]
            elif j == 1:
                max_sepal_width = column[length -1]
                min_sepal_width = column[0]
            elif j == 2:
                max_petal_length = column[length-1]
                min_petal_length = column[0]
            else:
                max_petal_width = column[length-1]
                min_petal_width = column[0]
        #for i in range(length):



    diff_sepal_length = max_sepal_length - min_sepal_length
    diff_sepal_width = max_sepal_width - min_sepal_width
    diff_petal_length = max_petal_length - min_petal_length
    diff_petal_width = max_petal_width - min_petal_width





