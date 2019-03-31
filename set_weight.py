# -*- coding:utf8 -*-
# @TIME     : 2019/3/17 10:17
# @Author   : SuHao
# @File     : set_weight.py

import math
import numpy as np
from sklearn import preprocessing


def gaussian(dist, a=1, b=0, c=0.8):
    dist = (dist - np.mean(dist))/np.std(dist)    #高斯方法采用正态归一化
    return a * math.e ** (-(dist - b) ** 2 / (2 * c ** 2))


def reverse(dist, const=1):
    min_max_scaler = preprocessing.MinMaxScaler()
    dist = min_max_scaler.fit_transform(dist)   #采用库函数自带的最大最小归一化
    return 1 / (dist + const)


def reverse_weight(neighbor, label_num):
    # 此处neighbor存放着近邻点的距离和对应的标签
    label = neighbor[:, 1].astype(np.int8)
    prob = np.zeros((label_num, ))
    reverse_dist = reverse(neighbor[:, 0])
    for i in range(label_num):
        prob[i] = np.sum((label == i) * reverse_dist)
    prob = prob / np.sum(reverse_dist)
    result = np.argmax(prob)
    return prob, result


def uniform_weight(neighbor, label_num):
    label = neighbor[:, 1].astype(np.int8)
    prob = np.bincount(
        label, minlength=label_num) / neighbor.shape[0]
    result = np.argmax(prob)
    return prob, result


def gaussian_weight(neighbor, label_num):
    label = neighbor[:, 1].astype(np.int8)
    prob = np.zeros((label_num, ))
    gaussian_dist = gaussian(neighbor[:, 0])
    for i in range(label_num):
        prob[i] = np.sum((label == i) * gaussian_dist)
    prob = prob / np.sum(gaussian_dist)
    result = np.argmax(prob)
    return prob, result
