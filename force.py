# -*- coding:utf8 -*-
# @TIME     : 2019/3/17 10:08
# @Author   : SuHao
# @File     : force.py


# def distsort(a, k, option=False):
#     '''
#     \param: a:array, 有两列第一列是距离，第二列是标签
#     \param: k:排序的个数
#     '''
#     if a.shape[1] != 2:
#         return
#     x = copy.copy(a)
#     y = np.zeros((k,2))
#     for i in range(k):
#         tmp = np.argmin(x[:,0])
#         y[i,:] = x[tmp,:]
#         x = np.delete(x, tmp, 0)
#     if option:
#         return y
#     else:
#         return y[:, 1]

import numpy as np
import dist_formular as metric
import copy



def distsort(a, k, option=True):
    '''
    \param: a:array
    \param: k:排序的个数
    '''
    if a.shape[1] != 2:
        return
    x = copy.copy(a)
    if k == 0:
        k = len(x)
    y = x[0:k, :]  # 先选取前k个存在队列里，之后要进行排序

    for i in range(k):
        for j in range(k - 1 - i):
            if y[j, 0] >= y[j + 1, 0]:
                y[[j, j + 1], :] = y[[j + 1, j], :]  # 冒泡排序

    for i in range(np.shape(x)[0] - k):
        for j in range(k - 1):
            if x[k + i, 0] >= y[k - 1, 0]:
                break
            elif x[k + i, 0] < y[0, 0]:
                y = np.insert(y, 0, x[k + i, :], axis=0)
                break
            elif x[k + i, 0] < y[k - 1 - j, 0] and x[k + i, 0] >= y[k - 2 - j, 0]:
                y = np.insert(y, k - 1 - j, x[k + i, :], axis=0)
                break
    if option:
        return y[0:k, :]
    else:
        return y[0:k, 1]


def force_research(dist_form, train_X, train_Y, test_x, k, p=None, V=None):
    '''
    :param dist_form: lambda表达式，计算距离的公式
    :param train_X: 训练数据
    :param train_Y: 训练数据标签
    :param test_x: 测试数据，只有一个点
    :param test_y: 测试数据的标签
    :param k: 近邻数
    :return: 返回对应近邻点的标签
    '''
    test_x = np.array([test_x])
    if dist_form == 'euclidean':
        dist = metric.euclidean(train_X.T, test_x)
    elif dist_form == 'manhattan':
        dist = metric.manhattan(train_X.T, test_x)
    elif dist_form == 'chebyshev':
        dist = metric.chebyshev(train_X.T, test_x)
    elif dist_form == 'minkowski':
        dist = metric.minkowski(train_X.T, test_x, p)
    elif dist_form == 'wminkowski':
        dist = metric.wminkowski(train_X.T, test_x, p, V)
    elif dist_form == 'seuclidean':
        dist = metric.seuclidean(train_X.T, test_x, V)
    elif dist_form == 'manhalanobis':
        dist = metric.manhalanobis(train_X.T, test_x, V)
    elif dist_form == 'correlation':
        dist = metric.correlation(train_X.T, test_x)
    elif dist_form == 'chi_square':
        dist = metric.chi_square(train_X.T, test_x)
    elif dist_form == 'intersection':
        dist = metric.intersection(train_X.T, test_x)
    elif dist_form == 'bhattacharyya':
        dist = metric.bhattacharyya(train_X.T, test_x)

    dist = dist[:, 0]
    table = np.array([dist, train_Y]).T  # 将计算结果和标签组合在一起
    neighbor = distsort(table, k)
    return neighbor
