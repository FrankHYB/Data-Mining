import csv
import sys
import math
import operator
import numpy as np

num_clusters_easy = 2;
num_clusters_hard = 4;

class TwoDimNode:

    def __init__(self, data, i, k):
        self.id = data['ID'][i]
        self.x1 = data['X.1'][i]
        self.x2 = data['X.2'][i]
        self.actual_cluster = data['cluster'][i]

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
        self.quality = data['quality'][i]
        self.cla = data['class'][i]

    def eulid(self, other):
        x = [self.fx_acidity, self.vol_acidity, self.citric_acid, self.resid_sugar, self.chlorides, self.free_sulf_d /
             self.tot_sulf_d, self.density, self.pH, self.sulph, self.alcohol]
        y = [other.fx_acidity, other.vol_acidity, other.citric_acid, other.resid_sugar, other.chlorides, other.free_sulf_d /
             other.tot_sulf_d, other.density, other.pH, other.sulph, other.alcohol]
        total = sum(math.pow(element1-element2, 2) for element1, element2 in zip(x,y))
        res = math.sqrt(total)
        return round(res, 5)





