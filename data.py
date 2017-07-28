import numpy as np
from numpy import genfromtxt

path = "data.csv"


def readData ():
    dataSet = genfromtxt(path, delimiter=',')
    m, n = np.shape(dataSet)
    trainData = np.ones((m, n))
    trainData[:, 1:] = dataSet[:, :-1]
    trainLabel = dataSet[:, -1]

    return trainData, trainLabel
