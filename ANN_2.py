# coding:utf-8
#两层神经网络
import numpy as np
#Sigmoid 函数可以将任何值都映射到一个位于 0 到  1 范围内的值。
def nonlin(x,deriv=False):
    if(deriv==True):#通过 “nonlin” 函数体还能得到 sigmod 函数的导数（当形参 deriv 为 True 时）
        return x*(1-x)
    return 1/(1+np.exp(-x))


X = np.array([ [0,0,1],
               [0,1,1],
               [1,0,1],
               [1,1,1] ])


y = np.array([[0,1,1,1]]).T



np.random.seed(1)

#零号突触”（即“输入层-第一层隐层”间权重矩阵）。
syn0 = 2*np.random.random((3,1)) - 1

for j in xrange(60000):#神经网络训练
    # forward propagation
    l0 = X#网络第一层 网络输入层
    l1 = nonlin(np.dot(l0,syn0))#网络第二层 隐藏层
    #这是神经网络的前向预测阶段
    # how much did we miss?
    l1_error = y - l1
 
    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)
 
    # update weights
    syn0 += np.dot(l0.T,l1_delta)#dot 若 x 和 y 为向量，则进行点积操作；若均为矩阵，则进行矩阵相乘操作；若其中之一为矩阵，则进行向量与矩阵相乘操作。
print "Output After Training:"
print l1