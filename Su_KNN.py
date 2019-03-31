# -*- coding:utf8 -*-
# @TIME     : 2019/3/6 15:39
# @Author   : SuHao
# @File     : Su_KNN.py

import numpy as np

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:58:57 2019

@author: Hao
"""

import numpy as np
import force
import Su_kdtree as kd
import set_weight as weight


class classifier:

    def __init__(
            self,
            research='force',
            dist_form='euclidean',
            weight='uniform',
            k=5,
            p=None,
            V=None):
        '''
        :param research: 搜索方法，两种选择：force和kdtree
        :param dist_form: 距离类型
        :param weight:计算权重的类型
        :param p:计算距离需要用到的参数
        :param V:计算距离需要用到的参数
        :param k:近邻数
        '''
        self.research = research
        self.dist_form = dist_form
        self.weight = weight
        self.p = p
        self.V = V
        self.k = k

    def train(self, train_X, train_Y, prior):
        '''
        :param train_X: training data, which size is M*N, where M is number of attribution and N is num of sample.
        :param trian_Y: corresponding label, string or value
        :param prior: 先验概率，可以指定
        :return:
        '''
        self.train_X = train_X
        self.train_Y = train_Y
        self.dimensions = train_X.shape[0]    #数据的属性个数，也可以说是维数
        if prior is None:                     #如果prior is not None，则认为要制指定先验概率
            self.prior = np.bincount(train_Y) / len(train_Y)
        else:
            self.prior = prior
        self.label_num = np.shape(self.prior)[0]  # label必须从0开始，否则此会出错

        # 选择使用的搜索算法，如果使用kdtree搜索，在训练数据部分进行构建kdtree
        if self.research == 'kdtree':
            table = np.concatenate((train_X.T, np.array([train_Y]).T), axis=1)
            self.tree = kd.create(
                table.tolist(), self.dimensions + 1, 0, None)

    def predict(self, test_X):
        '''
        predict_result: 存放预测结果
        predict_prob：存放置信度
        neighbor_dist: 存放最近点的距离和对应的标签
        1.使用bincount函数统计最近邻点中各个标签的占比，并将占比最大的标签作为预测结果。
        2.使用argmax函数求得占比最大对应的标签
        '''
        predict_result = np.zeros((np.shape(test_X)[1], ))
        predict_prob = np.zeros((self.label_num, np.shape(test_X)[1], ))
        neighbor_dist = np.zeros(
            (test_X.shape[1], self.k * 2))  # 用于存放所有测试样本近邻距离和对应标签
        if self.research == 'force':
            for index, test_x in enumerate(test_X.T):
                neighbor = force.force_research(
                    train_X=self.train_X,
                    train_Y=self.train_Y,
                    dist_form=self.dist_form,
                    test_x=test_x,
                    k=self.k,
                    p=self.p)
                neighbor_dist[index, 0:self.k] = neighbor[:, 0]
                neighbor_dist[index, self.k:self.k * 2] = neighbor[:, 1]

                if self.weight == 'gaussian':
                    predict_prob[:, index], predict_result[index] = weight.gaussian_weight(neighbor, self.label_num)
                elif self.weight == 'reverse_weight':
                    predict_prob[:, index], predict_result[index] = weight.reverse_weight(neighbor, self.label_num)
                else:
                    predict_prob[:, index], predict_result[index] = weight.uniform_weight(neighbor, self.label_num)

            return predict_result, predict_prob, neighbor_dist
        elif self.research == 'kdtree':
            for index, test_x in enumerate(test_X.T):
                neighbor = self.tree.kdtree_research(test_x.tolist(), self.k)
                neighbor_dist[index, 0:self.k] = neighbor[:, 0]
                neighbor_dist[index, self.k:self.k * 2] = neighbor[:, 1]
                neighbor = neighbor[:, 1].astype(np.int8)
                predict_prob[:, index] = np.bincount(
                    neighbor, minlength=self.label_num) / self.k
                predict_result[index] = np.argmax(predict_prob[:, index])
            return predict_result, predict_prob, neighbor_dist

    def test(self, test_X, test_Y):
        predict_result, predict_prob, _ = self.predict(test_X)
        accurate = np.sum(predict_result == test_Y) / np.shape(test_Y)[0]
        return accurate
