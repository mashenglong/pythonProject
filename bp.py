from numpy import *
import numpy as np


def sigmiods(z):  # 定义激活函数,针对于数组元素
    a = []
    for each in z:
        b = 1 / (1 + math.exp(-each[0]))
        a.append(b)
    return a


def forword(X, W, V, B1, B2):  # 定义前向传播
    net1 = W.T * X + B1
    H = matrix(sigmiods(np.array(net1))).T
    net2 = V.T * H + B2
    pred_y = matrix(sigmiods(np.array(net2))).T
    return pred_y, H


def back_p(Y, pred_y, H, V, aph, W):
    Errorterm = 0.5 * (Y - pred_y).T * (Y - pred_y) # 定义loss函数，使用均方差
    # 计算输出单元的误差
    a1 = multiply(pred_y - Y, pred_y)  # 点乘对应元素相乘
    a2 = multiply(a1, 1 - pred_y)
    Verror = H * a2.T
    # 计算隐藏单元的误差
    Werror = X * (multiply(multiply(H, 1 - H), (V * a2))).T
    # 更新权重
    Vupdata = V - aph * Verror
    Wupdata = W - aph * Werror
    return Vupdata, Wupdata, Errorterm


if __name__ =='__main__':
    X = matrix([0.05, 0.10]).T
    Y = matrix([0, 1]).T
    W = matrix([[0.15, 0.20], [0.25, 0.30]])
    B1 = matrix([0.1, 0.1]).T
    V = matrix([[0.40, 0.45], [0.50, 0.55]])
    B2 = matrix([0.2, 0.2]).T
    # 随机生成参数
    # np.random.seed(0)
    # W = matrix(np.random.normal(0,1,[2,2]))
    # B1 = matrix(np.random.normal(0, 1, [1, 2]))
    # V = matrix(np.random.normal(0, 1, [2, 2]))
    # B2 = matrix(np.random.normal(0, 1, [1, 2]))
    aph = 0.5
    n = 10  # 迭代次数
    for i in range(n):
        # 前向算法
        pred_y, H = forword(X, W, V, B1, B2)
        # 反向传播
        Vupdate, Wupdate, Errorterm = back_p(Y, pred_y, H, V, aph, W)
        W, V = Wupdate, Vupdate
    print('迭代次数：%d' % n)
    print('预测值：')
    print(pred_y)
    print('更新的权重V：')
    print(Vupdate)
    print('更新的权重W:')
    print(Wupdate)
    print('损失值：')
    print(Errorterm)
