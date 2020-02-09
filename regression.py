import numpy as np
from numpy.linalg import det
from numpy.linalg import inv
from numpy import mat
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd
import math

class LinearRegression:

    def _ini_(self):
        pass

    def train(self,x_train,y_train):
        x_mat = mat(x_train).T
        y_mat = mat(y_train).T
        [m, n] = x_mat.shape
        x_mat = np.hstack((x_mat, mat(np.ones((m, 1)))))
        self.weight = mat(random.rand(n + 1, 1))
        if det(x_mat.T * x_mat) == 0:
            print('the det of xTx is equal to zero.')
            return
        else:
            self.weight = inv(x_mat.T * x_mat) * x_mat.T * y_mat
        return self.weight

    def plot_lr(self, x_train, y_train):
        x_min = x_train.min()
        x_max = x_train.max()
        print(self.weight) #2*2的矩阵，self.weight[0]为θ，self.weight[2]为b
        y_min = self.weight[0] * x_min + self.weight[1]
        y_max = self.weight[0] * x_max + self.weight[1]
        plt.scatter(x_train, y_train)
        plt.plot([x_min, x_max], [y_min[0, 0], y_max[0, 0]], '-g')
        plt.show()

    def ridge_regression(self, x_train, y_train, lam=0.2):
        x_mat = mat(x_train).T
        [m, n] = np.shape(x_mat)
        x_mat = np.hstack((x_mat, mat(np.ones((m, 1)))))
        y_mat = mat(y_train).T
        self.weight = mat(random.rand(n + 1, 1))
        xTx = x_mat.T * x_mat + lam * mat(np.eye(n)) #np.eye 单位矩阵
        #加入惩罚项lam，防止xTx不是满秩矩阵而无法求逆矩阵
        self.weight = xTx.I * x_mat.T * y_mat
        return self.weight

    def lasso_regression(self, x_train, y_train, lam=0.01, itr_num=100):
        x_mat = mat(x_train).T
        y_mat = mat(y_train).T
        [m, n] = np.shape(x_mat)
        #对x_mat进行中心化和标准化
        # x_mat = (x_mat - x_mat.mean(axis=0)) / x_mat.std(axis=0)
        # y_mat = (y_mat - y_mat.mean(axis=0)) / y_mat.std(axis=0)
        x_mat = np.hstack((x_mat, mat(np.ones((m, 1)))))
        self.weight = mat(random.rand(n + 1, 1))

        for it in range(itr_num):
            for k in range(n):
                z_k = (x_mat[:,k].T * x_mat[:,k])[0,0]
                # z_k = 0
                # for i in range(m):
                #     z_k += x_mat[i,k].T * x_mat[i,k]
                # print(z_k)
                p_k = 0
                for i in range(m):
                    p_k += x_mat[i,k] * (y_mat[i,0] - sum([x_mat[i,j] * self.weight[j,0] for j in range(n) if j != k]))
                if p_k < -lam / 2:
                    w_k = (p_k + lam / 2) / z_k
                elif p_k > lam / 2:
                    w_k = (p_k - lam / 2) / z_k
                else:
                    w_k = 0
                self.weight[k,0] = w_k
        return self.weight

    def lwlr(self,x,X,Y,k):
        # print(X)
        z = X
        X = np.mat(X)
        Y = np.mat(Y)
        X = X.T
        Y = Y.T
        m,n = np.shape(X)
        print(m)
        X = np.hstack((np.mat(np.ones((m, 1))),X))
        print(X.shape)
        # print(m)
        # 创建针对x的权重矩阵
        W = np.matrix(np.zeros((m, m)))
        for i in range(m):
            # print(X[i])
            xi = np.array(z[i])
            # print(xi)
            x = np.array(x)
            W[i, i] = math.exp((np.linalg.norm(x - xi))/(-2*k**2))
        # 获取此点相应的回归系数
        
        # print(Y.shape)
        # print(X.shape)
        # print(W.shape)
        xWx = X.T*W*X
        # print(xWx.shape)
        if np.linalg.det(xWx) == 0:
            print('xWx is a singular matrix')
            return
        self.weight = xWx.I*X.T*W*Y
        return self.weight





def main():
    data = pd.read_csv("E:/coding-python/machine-learning/test.csv")
    data = data / 30
    x_train = data['x'].values
    y_train = data['y'].values
    # print(x_train)
    regression =  LinearRegression()
    # 基础线性回归
    # regression.train(x_train, y_train)
    # regression.plot_lr(x_train, y_train)
    #岭回归
    # regression.ridge_regression(x_train,y_train)
    # regression.plot_lr(x_train, y_train)
    #lasso回归
    # regression.lasso_regression(x_train,y_train)
    # regression.plot_lr(x_train,y_train)
    #局部加权线性回归
    regression.lwlr(0.5, x_train, y_train, 0.1)
    regression.plot_lr(x_train, y_train)


if __name__ == '__main__':
    main()