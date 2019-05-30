import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def Judge(X, y, w):
    n = X.shape[0]
    num = np.sum(X.dot(w) * y > 0)
    return num == n

def preprocess(data):
    n, d = data.shape
    X = data[:,:-1]
    y = data[:,-1]
    X = np.c_[np.ones(n), X]

    return X, y

def count(X, y, w):
    """
    统计错误点的数量
    
    """
    return np.sum(X.dot(w) * y < 0)


def Pocket_PLA(X, y, eta=1, iteration=1000):
    n, d = X.shape
    
    #初始化w
    w = np.zeros(d)
    
    
    #记录迭代次数
    iters = 0
    
    #先跑完全部的数据一遍，获得一个比较接近的w
    for i in range(n):
        if X[i,:].dot(w) * y[i] <= 0 and iters < iteration:
            w += eta * y[i] * X[i,:]
            iters += 1
    print('iteration{}'.format(iters)+',w:',w)
    best_count = count(X, y, w)
    best_w = w
    while Judge(X, y, w) == 0 and iters < iteration:
        for k in range(n):
            j = np.random.randint(0, n)
            if X[j,:].dot(w) * y[j] <= 0:           
                w += eta * y[j] * X[j,:]
                iters += 1
                #若得到了改进，则更新w和错误率
                if count(X, y, w) <= best_count:
                    best_w = w
                    best_count = count(X, y, w)
                    print('第{}次迭代中得到了改进，错误率为{}'.format(iters,best_count/n)+',w:',w)
                break
    #返回pocket中错误率最低的w  
    return best_w, iters, best_count/n    
