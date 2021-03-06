import csv
import sys
import math
import operator
import numpy as np
import random

num_clusters_easy = 2;
num_clusters_hard = 4;
initial_centroid_easy = []
cluster_easy =[]
initial_centroid_hard = []
cluster_hard = []
initial_centroid_wine = []
cluster_wine = []
two_dim_easy = 'TwoDimEasy.csv'
two_dim_hard = 'TwoDimHard.csv'
wine = 'wine.csv'
exist_counter = 0

class TwoDimNode:

    def __init__(self, data, i):
        self.id = data['ID'][i]
        self.x1 = data['X.1'][i]
        self.x2 = data['X.2'][i]
        self.actual_cluster = data['cluster'][i]
        self.prob = 0
        self.predict = 0
        self.centroid = None
        self.sc = 0

    def eulid(self, other):
        x = [self.x1, self.x2]
        y = [other.x1, other.x2]
        total = sum(math.pow(element1-element2, 2) for element1, element2 in zip(x,y))
        res = math.sqrt(total)
        return round(res, 5)

    def equals(self,other):
        #if self.x1 == other.x1 and self.x2 == other.x2:
        if (str(self.x1) == str(other.x1)) and (str(self.x2) == str(other.x2)):
            return True
        else:
            return False



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
        self.quality = data['quality'][i]
        self.density = data['density'][i]
        self.pH = data['pH'][i]
        self.sulph = data['sulph'][i]
        self.alcohol = data['alcohol'][i]
        if data['class'][i] == 'Low':
            self.actual_cluster = '1'
        else:
            self.actual_cluster = '2'
        self.prob = 0
        self.predict = 0
        self.centroid = None
        self.sc = 0

    def eulid(self, other):
        x = [self.fx_acidity, self.vol_acidity, self.citric_acid, self.resid_sugar, self.chlorides, self.free_sulf_d,
             self.tot_sulf_d, self.density, self.pH, self.sulph, self.alcohol]
        y = [other.fx_acidity, other.vol_acidity, other.citric_acid, other.resid_sugar, other.chlorides, other.free_sulf_d,
             other.tot_sulf_d, other.density, other.pH, other.sulph, other.alcohol]
        total = sum(math.pow(element1-element2, 2) for element1, element2 in zip(x,y))
        res = math.sqrt(total)
        return round(res, 5)

    def equals(self, other):
        if (self.fx_acidity == other.fx_acidity) and (self.vol_acidity == other.vol_acidity) and \
                (self.citric_acid == other.citric_acid) and (self.resid_sugar == other.resid_sugar) \
                and(self.chlorides == other.chlorides) and (self.free_sulf_d == other.free_sulf_d)and \
                (self.tot_sulf_d == other.tot_sulf_d) and self.density == other.density and \
                        self.pH == other.pH and self.sulph == other.sulph and self.alcohol == other.alcohol:
            return True
        else:
            return False



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
    with open(filename, 'w') as f:
        f.write(header)
        for node in nodes:
            f.write(str(node.id) + ',' + str(node.predict) + ',' + str(node.actual_cluster))
            f.write('\n')

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
            listOfCentroid.append(construct_centroid(points))



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
            listOfCentroid.append(nodes[arr[0]])


    #else:
    return listOfCentroid



#Calculate new centroid
#return list of new centroid
def point_avg(nodes,file_cluster,one_centroid = None, cluster_num = '0'):
    cluster = []
    if(len(nodes) < 500):
        #Calculate the sum for each attribute
        sum_x1 = []
        sum_x2 = []
        for n in nodes:
            if n.centroid == one_centroid or n.actual_cluster == cluster_num:
                sum_x1.append(n.x1)
                sum_x2.append(n.x2)
                cluster.append(n)

        #Average
        if len(sum_x1) == 0 or len(sum_x2) == 0:
            print 'randomly select a point with attributes with zero, please re-run'
            sys.exit(0)
        new_cen = [sum(sum_x1)/len(sum_x1),sum(sum_x2)/len(sum_x2)]
    else:
        #Calculate the sum for each attribute
        sum_fx =[]
        sum_vol = []
        sum_cit =[]
        sum_res =[]
        sum_chl =[]
        sum_free =[]
        sum_tot =[]
        sum_den =[]
        sum_ph =[]
        sum_sul =[]
        sum_alc =[]
        for n in nodes:
            if n.centroid == one_centroid or n.actual_cluster == cluster_num:
                sum_fx.append(n.fx_acidity)
                sum_vol.append(n.vol_acidity)
                sum_cit.append(n.citric_acid)
                sum_res.append(n.resid_sugar)
                sum_chl.append(n.chlorides)
                sum_free.append(n.free_sulf_d)
                sum_tot.append(n.tot_sulf_d)
                sum_den.append(n.density)
                sum_ph.append((n.pH))
                sum_sul.append(n.sulph)
                sum_alc.append(n.alcohol)
                cluster.append(n)

        #Average
        length = len(cluster)
        new_cen = [sum(sum_fx)/length,sum(sum_vol)/length,sum(sum_cit)/length,sum(sum_res)/length,sum(sum_chl)/length,
                sum(sum_free)/length,sum(sum_tot)/length,sum(sum_den)/length,sum(sum_ph)/length,sum(sum_sul)/length,
                   sum(sum_alc)/length]

    file_cluster.append(cluster)
    return new_cen

#Compare whether or not a list of cendroids are the same
def compare_list_of_centroid(old, new):
    for i in range(len(old)):
        if not new[i].equals(old[i]):
            return False

    return True


#K means algorithm
def k_means(nodes,k,ini_centroid,cluster):

    cnt = 0
    while True:
        #Assign all points to the closest centroid
        for n in nodes:
            dis = sys.float_info.max
            for i in range(k):
                new_dis = n.eulid(ini_centroid[i])
                if new_dis < dis:
                    dis = new_dis
                    n.centroid = ini_centroid[i]

        #Update centroid
        new_centroid= []
        for i in range(k):
            one_centroid_num = point_avg(nodes,cluster, ini_centroid[i])
            new_centroid.append(construct_centroid(one_centroid_num))

        if compare_list_of_centroid(ini_centroid,new_centroid):
            for i in range(len(cluster)-k):
                del cluster[0]

            break
        else:
            for i in range(len(ini_centroid)):
                ini_centroid[i] = new_centroid[i]

        cnt = cnt + 1



#convert a list number to dictionary in order to get new node
def construct_centroid(points):
    new_dict = {'ID':'0'}
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
        new_dict.update({'quality': ['0']})
        centroid = WineNode(new_dict,0)
    return centroid


def predict_cluster(cluster,nodes,k, actual):
    for i in range(k):
        if i+1 > actual:
            for node in cluster[i]:
                node.predict = str(i+1)
            continue
        counter = {}

        for node in cluster[i]:
            if node.actual_cluster in counter:
                counter[node.actual_cluster] = counter[node.actual_cluster] + 1
            else:
                counter.update({node.actual_cluster:1})
        max_key = max(counter, key=lambda k: counter[k])
        for node in cluster[i]:
            node.predict = max_key



#Calculate SSE
def compute_sse(nodes, centroids):
    sse = []
    for i in range(len(centroids)):
        every_sse = 0
        for node in nodes:
            if node.centroid != centroids[i]:
                continue
            every_sse += math.pow(node.eulid(node.centroid), 2)
        sse.append(every_sse)
    return sum(sse), sse

def compute_ssb(nodes, cluster):
    ssb = 0
    if isinstance(nodes[0], TwoDimNode):
        sum_x1 = []
        sum_x2 = []
        for node in nodes:
            sum_x1.append(node.x1)
            sum_x2.append(node.x2)
        points = [sum(sum_x1) / len(sum_x1), sum(sum_x2) / len(sum_x2)]
        overall_centre = construct_centroid(points)
        for c in cluster:
            ssb += len(c) * math.pow(overall_centre.eulid(c[0].centroid), 2)
    else:
        sum_fx =[]
        sum_vol = []
        sum_cit =[]
        sum_res =[]
        sum_chl =[]
        sum_free =[]
        sum_tot =[]
        sum_den =[]
        sum_ph =[]
        sum_sul =[]
        sum_alc =[]
        for n in nodes:
            sum_fx.append(n.fx_acidity)
            sum_vol.append(n.vol_acidity)
            sum_cit.append(n.citric_acid)
            sum_res.append(n.resid_sugar)
            sum_chl.append(n.chlorides)
            sum_free.append(n.free_sulf_d)
            sum_tot.append(n.tot_sulf_d)
            sum_den.append(n.density)
            sum_ph.append((n.pH))
            sum_sul.append(n.sulph)
            sum_alc.append(n.alcohol)
        points = [sum(sum_fx) / len(sum_fx), sum(sum_vol)/ len(sum_vol), sum(sum_cit)/len(sum_cit), sum(sum_res) / len(sum_res),
                  sum(sum_chl) / len(sum_chl), sum(sum_free) / len(sum_free), sum(sum_tot) / len(sum_tot), sum(sum_den)/ len(sum_den),
                  sum(sum_ph)/len(sum_ph), sum(sum_sul) / len(sum_sul), sum(sum_alc)/len(sum_alc)]
        overall_centre = construct_centroid(points)
        for c in cluster:
            ssb += len(c) * math.pow(overall_centre.eulid(c[0].centroid), 2)

    return ssb


def compute_slicoef(nodes, clusters):

    for cluster in clusters:
        for node in cluster:
            sc_a = 0
            min_scb = sys.float_info.max
            #compute a
            for other in cluster:
                sc_a += node.eulid(other)
            #compute b
            for other_cluster in clusters:
                sc_b = 0
                if cluster == other_cluster:
                    continue
                for n in other_cluster:
                    sc_b += node.eulid(n)
                if min_scb > sc_b / len(other_cluster):
                    min_scb = sc_b / len(other_cluster)
            avg_sca = sc_a / (len(cluster) - 1)
            node.sc = (min_scb - avg_sca) / max(min_scb, avg_sca)
    sli_clusters = []
    overall_sli = 0
    for i in range(len(clusters)):
        sum_sli = 0
        for node in clusters[i]:
            sum_sli += node.sc
        sli_clusters.append(sum_sli / len(clusters[i]))
        overall_sli += sum_sli
    return sli_clusters, overall_sli / len(nodes)



def confusion_matrix(nodes,cluster):
    truepos = []
    trueneg = []
    falsepos = []
    falseneg = []
    for i in range(len(cluster)):

        for node in cluster[i]:
            if(node.actual_cluster == '1' and node.predict == '1'):
                truepos.append(node)
            elif(node.actual_cluster == '1' and node.predict == '2'):
                falseneg.append(node)
            elif(node.actual_cluster == '2' and node.predict == '1'):
                falsepos.append(node)
            elif(node.actual_cluster == '2' and node.predict == '2'):
                trueneg.append(node)

    print 'tp = ' + str(len(truepos))
    print 'tn = ' + str(len(trueneg))
    print 'fp = ' + str(len(falsepos))
    print 'fn = ' + str(len(falseneg))
    acc = 1.0 * (len(truepos)+len(trueneg))/(len(nodes))
    print 'Accuracy = ' + str(acc)


def confusion_matrix_hard(nodes,cluster):
    mat = {}
    for i in range(len(cluster)):
        for node in cluster[i]:
            pair = repr([node.actual_cluster,node.predict])
            if mat.has_key(pair):
                mat[pair] = mat[pair] + 1
            else:
                mat.update({pair:1})

    print mat





if __name__ == '__main__':
    if len(sys.argv) == 2:
        K = int(sys.argv[1])
    else:
        print 'Please enter the number of clusters'
        sys.exit(0)

    #Read all files
    data_easy = readFile(two_dim_easy)
    data_hard = readFile(two_dim_hard)
    data_wine = readFile(wine)

    #Normalization
    for key, value in data_easy.items():
        if key == 'cluster' or key == 'ID' or key == 'quality':
            continue
        data_easy[key] = min_max_normalize(value)

    for key, value in data_hard.items():
        if key == 'cluster'or key == 'ID' or key == 'quality':
            continue
        data_hard[key] = min_max_normalize(value)

    data_wine['tot_sulf_d'] = map(float,data_wine['tot_sulf_d'])
    data_wine['sulph'] = map(float,data_wine['sulph'])
    data_wine['alcohol'] = map(float,data_wine['alcohol'])
    data_wine['vol_acidity'] = map(float,data_wine['vol_acidity'])



    for key, value in data_wine.items():
        if key == 'class'or key == 'ID' or key == 'quality':
            continue
        if key =='tot_sulf_d':
            for i in range(len(value)):
                if value[i] < 95:
                    value[i] = 1
                else:
                    value[i] = 10
        if key =='sulph':
            for i in range(len(value)):
                if value[i] < 0.58:
                    value[i] = 0.3
                if value[i] >= 0.74:
                    value[i] = 0.95


        if key == 'alcohol':
            for i in range(len(value)):
                if value[i] <= 9.9:
                    value[i] = 7
                if value[i] > 10.5:
                    value[i] = 13

        if key == 'vol_acidity':
            for i in range(len(value)):
                if value[i] <= 0.4:
                    value[i] = 0.2
                if value[i] >= 0.72:
                    value[i] = 0.92
        data_wine[key] = min_max_normalize(value)


    #Create nodes
    easy_nodes = []
    hard_nodes = []
    wine_nodes = []
    for i in range(len(data_easy['ID'])):
        easy_nodes.append(TwoDimNode(data_easy, i))

    for i in range(len(data_hard['ID'])):
        hard_nodes.append(TwoDimNode(data_hard, i))

    for i in range(len(data_wine['ID'])):
        wine_nodes.append(WineNode(data_wine, i))

    # Select initial centroids, and call the K means function
    #Note: 1: randomly select data points as initial centroids,
    #      2. randomly select a number between min and max of each attribute, and form initial centroid
    #      3.

    #TWODIM_EASY
    initial_centroid_easy = initial_centroid(data_easy, 1, K, easy_nodes)
    k_means(easy_nodes,K,initial_centroid_easy,cluster_easy)
    predict_cluster(cluster_easy,easy_nodes,K,2)

    overall_SSE_easy, easy_sse = compute_sse(easy_nodes, initial_centroid_easy)
    easy_ssb = compute_ssb(easy_nodes, cluster_easy)
    sli_cluster, overall = compute_slicoef(easy_nodes, cluster_easy)

    print 'Overall_SSE_easy = ' + str(overall_SSE_easy) +'\nSSE for each cluster: '
    print easy_sse
    print 'SSB = ' + str(easy_ssb)
    print 'Overall silhouette width = ' + str(overall) + '\nSilhouette width for each cluster: '
    print sli_cluster
    print 'Confusion Matrix'
    confusion_matrix_hard(easy_nodes,cluster_easy)


    #TWODIM_HARD
    initial_centroid_hard = initial_centroid(data_hard, 1, K, hard_nodes)
    k_means(hard_nodes,K,initial_centroid_hard,cluster_hard)
    predict_cluster(cluster_hard,hard_nodes,K,4)

    overall_SSE_hard, hard_sse = compute_sse(hard_nodes, initial_centroid_hard)
    hard_ssb = compute_ssb(hard_nodes, cluster_hard)
    sli_cluster, overall = compute_slicoef(hard_nodes, cluster_hard)

    print '\nOverall_SSE_hard =' + str(overall_SSE_hard) +'\nSSE for each cluster: '
    print hard_sse
    print 'SSB = ' + str(hard_ssb)
    print 'Overall silhouette width = ' + str(overall) + '\nSilhouette width for each cluster: '
    print sli_cluster
    print 'Confusion Matrix'
    confusion_matrix_hard(hard_nodes,cluster_hard)


    #WINE
    initial_centroid_wine = initial_centroid(data_wine, 3, K, wine_nodes)
    k_means(wine_nodes,K,initial_centroid_wine,cluster_wine)
    predict_cluster(cluster_wine,wine_nodes,K,2)

    overall_SSE_wine, wine_sse = compute_sse(wine_nodes, initial_centroid_wine)
    wine_ssb = compute_ssb(wine_nodes, cluster_wine)
    sli_cluster, overall = compute_slicoef(wine_nodes, cluster_wine)

    print '\nOverall_SSE_wine = ' + str(overall_SSE_wine) + '\nSSE for each cluster:'
    print wine_sse
    print 'SSB: ' + str(wine_ssb)
    print 'Overall silhouette width = ' + str(overall) + '\nSilhouette width for each cluster'
    print sli_cluster
    print 'Confusion Matrix'
    confusion_matrix(wine_nodes,cluster_wine)
    header = 'ID,predict_cluster,true_cluster\n'
    filename_easy = 'TwoDimEasyOutput.csv'
    filename_hard = 'TwoDimHardOutput.csv'
    filename_wine = 'WineOutput.csv'
    write_to_file(filename_easy,header,easy_nodes)
    write_to_file(filename_hard,header,hard_nodes)
    write_to_file(filename_wine,header,wine_nodes)



