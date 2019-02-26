import os.path

import random
import math
import time
import sys
import string
import shlex
import subprocess

import scipy.stats
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

confidence_cut = 0.7
normalize_by_sum = False


# function for string float conversion
def str_to_float(arr):
    t = []
    for i in range(0, len(arr)):
        t.append(float(arr[i]))

    return t


# function for list saving
def write_list(fn, list):
    f = open(fn, 'w')
    for n in list:
        f.write(n + " \n")

    f.close()

# tsv file reading function


def read_tsv(fn):
    data = []
    f = open(fn, 'r')
    f.readline()

    for line in f:
        sl = line.split()
        data.append(sl)

    return data


# function for mapping read data into features and target values
def get_map(data):
    dict = {
        'Yes': [1.0, 0.0],
        'No': [0.0, 1.0]
    }

    print("the representation of the labels")
    print(dict)

    mapped_data_x = []
    mapped_data_y = []

    for d in data:
        mapped_data_x.append(str_to_float(d[2:]))
        mapped_data_y.append(dict[d[0]])

    return dict, mapped_data_x, mapped_data_y


def get_name(data):
    name_list = []

    for d in data:
        name_list.append(d[1])

    return name_list

# select data by their names


def select_by_name(data, namelist):
    selected = []
    for n in namelist:
        for d in data:
            if n == d[1]:
                selected.append(str_to_float(d[2:]))
                break

    return np.array(selected)

# get peak data


def obtain_data_from_list(the_data, data_list):
    obtained_data = []
    for i in range(0, len(data_list)):
        obtained_data.append(the_data[data_list[i]])

    return np.array(obtained_data)

# function for getting random batches


def get_batch(x_data, y_data, s):
    lowest = 0
    highest = len(x_data)

    rand_list = np.random.randint(low=lowest, high=highest, size=s)
    return x_data[rand_list], y_data[rand_list]
