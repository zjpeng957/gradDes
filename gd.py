import numpy as np
import data
import matplotlib.pyplot as plt
import math


def sigmoid (x, theta):
    z = np.dot (x, theta)
    d, c = np.shape (z)
    h = np.reciprocal (np.ones ((d, 1)) + np.exp (np.negative (z)))
    return h


def batchGradientDescent (x, y, theta, alpha, max_N):
    xT = x.transpose ()
    for i in range (0, max_N):
        h = sigmoid (x, theta)
        loss = y - h
        gradient = np.dot (xT, loss)
        theta = theta + alpha * gradient
    return theta


def predict (x, theta):
    y = sigmoid (x, theta)
    w = y > 0.5
    return w


if __name__ == "__main__":
    t_data, t_label = data.readData ()
    t_label = np.array ([[0], [0], [1], [0], [0], [1],[0],[0],[0],[1],[1],[0],[1]])
    the = np.zeros ((3, 1))
    alp = 0.001
    MAX_N = 1000
    the = batchGradientDescent (t_data, t_label, the, alp, MAX_N)

    xp = np.array ([[1, 1, 2], [1, 5, 6], [1, 0, 0], [1, 32, 56]])
    result = predict (xp, the)
    plt.plot ([1, 2, 3, 4], result)
    print (result)
    plt.show ()
