import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正向显示负号

def Judge(X,y,w):
    """

    判别函数，判断所有数据是否分类完成
    """

    n = X.shape[0]
    num = np.sum(X.dot(w) * y > 0) 
    return num == n


def data(N, d, rnd):
    """
    生成N个d维的点，不包括1，x1+...+xd>0的点标记为+1，x1+...+xd<0的点标记为-1，
    rnd为随机数生成器，形式为rnd = np.random.RandomState(seed)
    """

    X = []
    
    w = np.ones(d)  #array([1., 1., 1.,... 1., 1.])
    while len(X) < N:
        x = rnd.uniform(-1, 1, d)     #-1到1之间的d个随机数
        if np.abs(x.dot(w)) > 0:     #x1+...+xd>=t或<=t
            X.append(x)
    X = np.array(X)
    y = 2 * (X.dot(w) > 0) - 1  #sign，为真时为1，假时为-1

    #添加第一个分量为1
    X = np.c_[np.ones((N,1)),X] #c_是左右拼接矩阵，r_是上下拼接

    return X,y

def f(N, d, rnd, r=1):
    """
    利用PLA更新，如果r=1，则顺序取点，否则随机取点
    """
    X,y = data(N, d, rnd)

    s = 0 #记录次数
    
    w = np.zeros(d + 1) #初始化w = [0,0,0]
    n = X.shape[0] #数据数量
    
    if r == 1:
        while Judge(X, y, w) == 0:
            for i in range(n):
                if X[i,:].dot(w) * y[i] <= 0:
                    w += y[i] * X[i,:]
                    s += 1

    else:
        while Judge(X, y, w) == 0:
            i = np.random.randint(0,N)
            if X[i,:].dot(w) * y[i] <= 0:
                w += y[i] * X[i,:]
                s += 1

    a = np.arange(-1,1,0.1)
    b = (a * w[1] + w[0]) / (-w[2])      #算法得到的直线
    c = - a                        #目标函数为x1+x2=0

    return a, b, c, X, y, s, w

def plot_helper(a, b, c, X, y, s, w):
    """
    作图函数
    
    """
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], c='r')
    plt.scatter(X[y == -1][:, 1], X[y == -1][:, 2], c='b')
    plt.plot(a, b, label='{}+{}x1+{}x2=0'.format(w[0],w[1],w[2]))
    plt.plot(a, c, label='x1+x2=0')
    plt.title('经过{}次迭代收敛'.format(s))
    plt.legend()
    plt.show()
