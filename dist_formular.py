# -*- coding:utf8 -*-
# @TIME     : 2019/3/16 12:01
# @Author   : SuHao
# @File     : dist_formular.py

from sklearn.neighbors import DistanceMetric
import numpy as np
from numba import jit

'''
行数为点的个数，列数为点的维度
'''


def euclidean(x, y):
    result = DistanceMetric.get_metric('euclidean')
    return result.pairwise(x, y)


def manhattan(x, y):
    result = DistanceMetric.get_metric('manhattan')
    return result.pairwise(x, y)


def chebyshev(x, y):
    result = DistanceMetric.get_metric('chebyshev')
    return result.pairwise(x, y)


def minkowski(x, y, p):
    result = DistanceMetric.get_metric('minkowski', p)
    return result.pairwise(x, y)


def wminkowski(x, y, p, w):
    result = DistanceMetric.get_metric('wminkowski', p, w)
    return result.pairwise(x, y)


def seuclidean(x, y, V):
    result = DistanceMetric.get_metric('seuclidean', V)
    return result.pairwise(x, y)


def manhalanobis(x, y, V):
    result = DistanceMetric.get_metric('manhalanobis', V)
    return result.pairwise(x, y)


'''以下距离公式参考
https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html
'''


@jit
def correlation(x, y):
    def single(a, b):
        return np.sum((a - np.mean(a)) * (b - np.mean(b))) / \
            np.sqrt(np.sum((a - np.mean(a))**2) * np.sum((b - np.mean(b))**2))
    z = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            z[i, j] = single(x[i, :], y[j, :])
    return z


def chi_square(x, y):
    def single(a, b):
        try:
            return np.sum((a - b)**2 / a)
        except ZeroDivisionError as e:
            print(e)
            return float("inf")  # 我不知道如何处理这里的异常
    z = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            z[i, j] = single(x[i, :], y[j, :])
    return z


def intersection(x, y):
    def single(a, b):
        return np.sum(np.minimum(a, b))
    z = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            z[i, j] = single(x[i, :], y[j, :])
    return z


def bhattacharyya(x, y):
    def single(a, b):
        N = len(a)
        return np.sqrt(
            1 -
            1 /
            np.sqrt(
                np.mean(a) *
                np.mean(b)) /
            N *
            np.sum(
                np.sqrt(
                    a *
                    b)))
    z = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            z[i, j] = single(x[i, :], y[j, :])
    return z
