# encoding:utf-8
# ********* 导入相应的模块***********
import math
import numpy as np
from numpy import *


# 激活函数
def sigmods(z):
    a = []
    for each in z:
        b = 1/(1+math.exp(-each[0]))
        a.append(b)
    return a


def maxout1(z):
    s = []
    for each in z:
        for m in each:
            s.append(max(m,0))
    return s


# 前向传播,返回预测值
def forwordmd(X,W,V,B1,B2):
    net1 = W.T*X+B1
    H = matrix(sigmods(np.array(net1))).T # 隐藏层单元
    net2 = V.T*H+B2
    pred_y = matrix(sigmods(np.array(net2))).T # 预测值
    return pred_y,H,net1,net2


# 反向传播,更新权重
def Bpaugorith(Y,pred_y,H,V,aph,W):
    Errorterm = 0.5*(Y-pred_y).T*(Y-pred_y)# 给出误差公式
    # 计算输出单元的误差项
    a1 = multiply(pred_y-Y,pred_y) # 矩阵对应元素相乘
    a2 = multiply(a1,1-pred_y)
    Verror = H*a2.T
    # 计算隐藏单元的误差项
    Werror = X*(multiply(multiply(H,1-H),(V*a2))).T
    # 更新权重
    Vupdate = V - aph*Verror
    Wupdate = W - aph*Werror
    return Vupdate,Wupdate,Errorterm


if __name__ =='__main__':
    X = matrix([0.05,0.10]).T
    Y = matrix([0,1]).T
    # 给出初始权重
    # W = matrix([[0.15,0.20],[0.25,0.30]])
    # B1 = matrix([0.1,0.1]).T
    # V = matrix([[0.40,0.45],[0.50,0.55]])
    # B2 = matrix([0.2,0.2]).T
    # 初始权重亦可随机生成
    # 随机生成参数
    np.random.seed(0)
    W = matrix(np.random.normal(0,1,[2,2]))
    B1 = matrix(np.random.normal(0, 1, [1, 2]))
    V = matrix(np.random.normal(0, 1, [2, 2]))
    B2 = matrix(np.random.normal(0, 1, [1, 2]))

    aph = 0.5 # 学习率
    # 阈值e，可根据需要自行更改
    e,m = 0.19,1
    pred_y, H, net1, net2 = forwordmd(X,W,V,B1,B2)  # 得到预测值和隐藏层值
    # 更新权重
    Vupdate, Wupdate, Errorvalue = Bpaugorith(Y,pred_y,H,V,aph,W)  # 得到更新的权重
    W,V = Wupdate,Vupdate
    while Errorvalue>e:
        # 激活前向算法
        pred_y, H, net1, net2 = forwordmd(X,W,V,B1,B2)  # 得到预测值和隐藏层值
    #     # 更新权重
        Vupdate, Wupdate, Errorvalue = Bpaugorith(Y,pred_y,H,V,aph,W)  # 得到更新的权重
        W, V = Wupdate, Vupdate
        m = m+1
    print('阈值e：%.2f' % e)
    print('更新权重:%d次' % m)
    print('预测值：')
    print(pred_y)
    print('更新的权重V：')
    print(Vupdate)
    print('更新的权重W:')
    print(Wupdate)
    print('损失值：')
    print(Errorvalue)
