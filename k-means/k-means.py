import numpy as np
from numpy.linalg import det
from numpy.linalg import inv
from numpy import mat
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd

class k_means:

    def __init__(self):
        pass

    def train(self,x,y,k,iter_num):
        m = mat(x).shape[1]
        #初始化聚类中心
        cluster = []
        np.random.seed(3)
        for i in range(k):
            cluster.append([np.random.randint(0, 80), np.random.randint(0, 80)])

        print(cluster)
        #初始化标签
        label = [0] * m

        for t in range(iter_num):
            for i in range(m):
                distance_min = 10000
                flag_min = 0
                for j in range(k):
                    dis = (cluster[j][0] - x[i]) ** 2 + (cluster[j][1] - y[i]) ** 2
                    if dis < distance_min:
                        distance_min = dis
                        flag_min = j
                    label[i] = flag_min

            for i in range(k):
                x_num = 0
                y_num = 0
                count = 0
                for j in range(m):
                    if label[j] == i:
                        x_num += x[j]
                        y_num += y[j]
                        count += 1
                # print(count)
                cluster[i][0] = x_num / count
                cluster[i][1] = y_num / count

        # print(label)
        label = np.array(label)
        index_0 = np.where(label == 0)
        plt.scatter(x[index_0], y[index_0], color='b')
        index_1 = np.where(label == 1)
        plt.scatter(x[index_1], y[index_1], color='r')
        index_2 = np.where(label == 2)
        plt.scatter(x[index_2], y[index_2], color='g')
        plt.show()




def main():
    data = np.loadtxt("E:/coding-python/machine-learning/k-means.csv")
    x = data[:,0]
    y = data[:,1]
    means = k_means()
    k = 3
    # plt.scatter(x,y)
    # plt.show()
    means.train(x,y,k,40)

if __name__ == '__main__':
    main()