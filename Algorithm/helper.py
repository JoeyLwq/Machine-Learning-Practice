import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正向显示负号

class Helper:
    def __init__(self):
        print('函数包括Judge(X,y,w)、data(N, d, data_range, w_f, seed)、plot_helper(X, y, w_g, w_f)')
    
    def Judge(self,X,y,w):
        """

        判别函数，判断所有数据是否分类完成
        """

        n = X.shape[0]
        num = np.sum(X.dot(w) * y > 0) 
        return num == n

    def data(self,N, d, data_range, w_f, seed):
        """
        生成N个d维的点，不包括1，x1+...+xd>0的点标记为+1，x1+...+xd<0的点标记为-1，
        rnd为随机数生成器，形式为rnd = np.random.RandomState(seed)
        """
        rnd = np.random.RandomState(seed)

        X = rnd.uniform(data_range[0], data_range[1], (N, d))     #-1到1之间的N个随机数,d维
        #添加第一列为1的分量
        X = np.c_[np.ones((N,1)),X] #c_是左右拼接矩阵，r_是上下拼接

        y = 2 * (X.dot(w_f) > 0) - 1  #sign，为真时为1，假时为-1

        return X,y


    def plot_helper(self,X, y, w_g, w_f=None):
        """
        作图函数，只能画2维的

        """
        x_range = np.arange(np.min(X[:,1]), np.max(X[:,1]), 0.1)
        plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], c='r')
        plt.scatter(X[y == -1][:, 1], X[y == -1][:, 2], c='b')
        
        g = (w_g[0] + w_g[1] * x_range) / -w_g[2]
        plt.plot(x_range, g, label='g')
        if w_f:
            f = (w_f[0] + w_f[1] * x_range) / -w_f[2]
            plt.plot(x_range, f, label='f')
        
        plt.legend()
        plt.show()
