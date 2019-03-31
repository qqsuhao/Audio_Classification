# -*- coding:utf8 -*-
# @TIME     : 2019/3/30 22:06
# @Author   : SuHao
# @File     : reload_and_classify.py

import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
import Su_KNN as KNN
import os
from scipy import stats


def reload_and_classify():
    saveprojectpath = '..\\仿真结果\\boy_and_girl_without_pre_amphasis_hilbert'
    savedata = saveprojectpath + '\\data'
    savepic = saveprojectpath + '\\pic'
    savetestdata = savedata + '\\' + 'test_data'
    savepreprocess = savedata + '\\' + 'preprocessing_result.csv'
    savetrainfeature = savedata + '\\' + 'feature_result_train.csv'
    savetestfeature = savedata + '\\' + 'test_data' + '\\' + 'feature_result_test'
    path = '..\\boy_and_girl'  # 数据集路径


    # 读取训练数据
    train = pd.read_csv(savetrainfeature, encoding='utf-8', header=None)
    train = train.values.astype('float32')
    train_X = train[:, 0:train.shape[1]-2]
    train_Y = train[:, -2]


    # 训练KNN分类器
    N = 5
    neigh = KNeighborsClassifier(
        n_neighbors=5,
        algorithm='brute',
        metric='euclidean')
    neigh.fit(train_X, train_Y)         # 使用我自己的KNN时，输入的特征矩阵列数代表样本数.使用库函数的KNN时正好相反。


    # 读取测试数据
    accurate = 0    # 预测正确的个数
    prob = []
    children = os.listdir(savetestdata)
    if len(children):
        for child in children:                # 为了防止多次测试导致测试样本数据文件夹里的文件重复，因此先删除
            test = pd.read_csv(savetestdata + '\\' + child, encoding='utf-8', header=None)
            test = test.values.astype('float32')
            test_X = test[:, 0:(test.shape[1]-2)]
            test_Y = test[0, -2]
            predictions = neigh.predict(test_X)
            predict_label, label_count = stats.mode(predictions)
            prob.append(label_count / test_X.shape[0])
            if test_Y == predict_label:
                accurate += 1
        accuracy = accurate / len(children)
    print(accuracy)
    print(list(prob))



    # params = {'dist_form': None,
    #           'test_X': test_X.T[:, 0:N],
    #           'test_Y': test_Y[0:N],
    #           'k': 5,
    #           'kind': 'heapsort'}
    # test = KNN.classifier('force', None)
    # test.train(train_X.T, train_Y, None)
    # pe = test.test(**params)
    # predict_result, predict_prob = test.predict(dist_form=None, k=5, test_X=test_X[:, 0:N], kind='heapsort')

