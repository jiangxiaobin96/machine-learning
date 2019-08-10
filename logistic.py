import numpy as np
from numpy.linalg import det
from numpy.linalg import inv
from numpy import mat
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd

class logistic:

    def __init__(self):
        pass

    def sigmoid(selt,z):
        return  1/(1+np.exp(-z))

    def output(self,x):
        g = np.dot(x,self.weight)
        return self.sigmoid(g)

    def compute_loss(self,x,y):
        m = x.shape[0]
        h = self.output(x)
        loss = -np.sum((y.T * np.log(h) + (1 - y).T * np.log((1 - h))))
        loss = loss / m
        dweight = x.T.dot((h - y)) / m
        return  loss,dweight

    def train(self,x,y,learn_rate=0.01,num_iters=5000):
        x = np.hstack((mat(np.ones((x.shape[0], 1))) , x))
        m,n = x.shape
        self.weight = 0.001 * np.random.randn(n,1).reshape((-1,1))
        loss = []

        for i in range(num_iters):
            error,dWeight = self.compute_loss(x,y)
            self.weight -= learn_rate * dWeight

            loss.append(error)

        return loss


def main():
    data = np.loadtxt("E:/coding-python/machine-learning/logistic.csv")
    x = data[:,[0,1]]
    y = data[:,2]
    x = mat(x)
    y = mat(y).T

    logist = logistic()
    #跟踪损失函数loss变化
    loss = logist.train(x,y)
    # plt.plot(loss)
    # plt.show()


    label = np.array(y)
    index_0 = np.where(label == 0)
    plt.scatter(x[index_0, 0].tolist(), x[index_0, 1].tolist(), marker='x', color='b', label='0', s=15)
    index_1 = np.where(label == 1)
    plt.scatter(x[index_1, 0].tolist(), x[index_1, 1].tolist(), marker='o', color='r', label='1', s=15)

    # show the decision boundary
    x1 = np.arange(x[:,0].min(),x[:,0].max())
    #θ_0+θ_1*x1+θ_2*x2=0
    x2 = (- logist.weight[0] - logist.weight[1] * x1) / logist.weight[2]
    plt.plot(x1, x2, color='black')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()