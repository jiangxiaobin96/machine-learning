import numpy as np
from numpy.linalg import det
from numpy.linalg import inv
from numpy import mat
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd

class SVM:

    def __init__(self):
        pass

    def clip(self,alpha,L,H):
        if alpha <  L:
            return L
        if alpha > H:
            return H
        else:
            return alpha

    #选择除i以外的另一个值
    def select_j(self,i,m):
        l = list(range(m))
        seq = l[:i] + l[i+1:]
        return random.choice(seq)

    def simple_smo(self,dataset,label,c,iter_num):
        [m,n] = dataset.shape
        alphas = np.zeros(m)
        alphas = mat(alphas)
        b = 0
        it = 0
        def f(x):
            x = np.mat(x).T #2x1矩阵
            ks = dataset * x
            wx = mat(np.array(alphas) * np.array(label.T)) * ks
            fx = wx + b
            return fx[0,0]

        while it < iter_num:
            pair_changed = 0
            for i in range(m):
                a_i,x_i,y_i = alphas.T[i],dataset[i],label[i]
                fx_i = f(x_i) #x_i 1x2矩阵
                E_i = fx_i - y_i
                j = self.select_j(i,m)
                a_j,x_j,y_j = alphas.T[j],dataset[j],label[j]
                fx_j = f(x_j)
                E_j = fx_j - y_j
                k_ii,k_jj,k_ij = np.dot(x_i,x_i.T),np.dot(x_j,x_j.T),np.dot(x_i,x_j.T)
                eta = k_ii + k_jj - 2 * k_ij
                if eta <= 0:
                    print("eta <= 0")
                    continue
                a_i_old,a_j_old = a_i,a_j
                a_j_new = a_j_old + y_j * (E_i - E_j) / eta
                if y_i != y_j:
                    L = max(0,a_j_old-a_i_old)
                    H = min(c,c + a_j_old - a_i_old)
                else:
                    L = max(0,a_i_old + a_j_old - c)
                    H = min(c,a_j_old + a_i_old)
                a_j_new = self.clip(a_j_new,L,H)
                a_i_new = a_i_old + y_i * y_j * (a_j_old - a_j_new)
                if abs(a_j_new - a_j_old) < 0.00001:
                    continue
                alphas.T[i],alphas.T[j] = a_i_new,a_j_new
                # 更新阈值b
                b_i = -E_i - y_i * k_ii * (a_i_new - a_i_old) - y_j * k_ij * (a_j_new - a_j_old) + b
                b_j = -E_j - y_i * k_ii * (a_i_new - a_i_old) - y_j * k_ij * (a_j_new - a_j_old) + b
                if 0 < a_i_new < c:
                    b = b_i
                elif 0 < a_j_new < c:
                    b = b_j
                else:
                    b = (b_i + b_j) / 2
                pair_changed += 1
            if pair_changed == 0:
                it += 1
            else:
                it = 0
        return alphas,b

    def get_w(self,alphas,dataset,label):
        alphas,dataset,label = np.array(alphas), np.array(dataset), np.array(label)
        yx = label.reshape(1, -1).T * np.array([1, 1]) * dataset
        w = np.dot(yx.T, alphas.T)
        return w.tolist()

def main():
    data = np.loadtxt("E:/coding-python/machine-learning/svm.csv")
    x = data[:,[0,1]]
    y = data[:,2]
    x = mat(x)
    y = mat(y).T

    svm = SVM()
    alphas,b = svm.simple_smo(x,y,0.6,40)

    label = np.array(y)
    index_0 = np.where(label == -1)
    plt.scatter(x[index_0, 0].tolist(), x[index_0, 1].tolist(), marker='x', color='b', label='0', s=15)
    index_1 = np.where(label == 1)
    plt.scatter(x[index_1, 0].tolist(), x[index_1, 1].tolist(), marker='o', color='r', label='1', s=15)


    w = svm.get_w(alphas, x, y)
    print(w)
    print(b)

    x1 = x[:,0].max()
    x2 = x[:,0].min()
    a1, a2 = w
    y1 = (- b[0] - a1[0] * x1) / a2[0]
    y2 = (- b[0] - a1[0] * x2) / a2[0]
    plt.plot([x1, x2], [y1[0,0], y2[0,0]])
    # 绘制支持向量
    # for i, alpha in enumerate(alphas):
    #     # print(alpha)
    #     if abs(alpha).any() > 1e-3:
    #         # print(x[i])
    #         p = x[i].T[0]
    #         q = x[i].T[1]
    #         plt.scatter([p], [q], s=150, c='none', alpha=0.7,
    #                     linewidth=1.5, edgecolor='#AB3319')
    plt.show()


if __name__ == '__main__':
    main()