import numpy as np
import data
import matplotlib.pyplot as plt


def batchGradientDescent (x, y, theta, alpha, max_N):
    xT = x.transpose()
    yT=y.transpose()
    for i in range(0, max_N):
        h = np.dot(x, theta)
        loss = h - y
        gradient = np.dot(xT, loss)
        theta = theta - alpha * gradient
    return theta


def predict(x, theta):
    y = np.dot(x, theta)
    return y


if __name__ == "__main__":
    t_data, t_label = data.readData()
    t_label=np.array([[16],[24],[38],[25],[21],[41]])
    the=np.zeros((3, 1))
    alp=0.001
    MAX_N=50
    the = batchGradientDescent(t_data, t_label, the, alp, MAX_N)

    xp=np.array([[1,1,2],[1,5,6],[1,0,0],[1,32,56]])
    result=predict(xp,the)
    plt.plot([1,2,3,4],result)
    print(result)
    plt.show()
