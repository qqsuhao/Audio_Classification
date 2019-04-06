# -*- coding:utf8 -*-
# @TIME     : 2019/3/30 22:06
# @Author   : SuHao
# @File     : reload_and_classify.py

import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import Su_KNN as KNN
import os
from scipy import stats
import visualization as visual
import random
import csv


def reload_and_classify(order,
                        saveprojectpath,
                        savedata,
                        savepic,
                        savetestdata,
                        savepreprocess,
                        savefeature,
                        path,
                        downsample_rate,
                        frame_length,
                        frame_overlap,
                        test_rate):
    '''
把数据集分为训练集和测试集
大体思路如下：
    1. 读取每一类标签文件下的音频文件数；
    2. 以每一类标签文件下的音频文件数为上限产生指定比例数量的随机数，对应的文件序号即为测试数据，将其放入一个列表。
        这个列表元素的个数为标签的个数，每个元素是一个按上述规定产生的随机数组
    3. 在进行特征提取时，在循环体里设置控制，将训练数据和测试书籍分开存放的两个文件。
    4. 在这个场景下，是一个集合对应一个标签。这对训练数据来说没有什么影响；但是对于测试数据，必须将每个集合分开存放。
    即使是属于同一类标签的集合，其元素也不能混合，否则会影响集合所属标签的判断。
    5. 专门创建以文件夹用来存放测试数据，以便之后读取数据更加方便。
'''
    sample_num = []
    labelname = os.listdir(savefeature)
    for i in range(len(labelname)):
        sample_num.append(len(os.listdir(savefeature + '\\' + labelname[i])))
    test_set = []      # 每一类标签文件下的测试样本对应序号
    for i in sample_num:
        test_set.append(
            random.sample(
                range(
                    0, i), int(
                    i * test_rate)))       # 随机挑选样本作为测试样本
    train_set = []
    for i in range(len(labelname)):
        tmp = list(range(0, sample_num[i]))
        [tmp.remove(j) for j in test_set[i]]
        train_set.append(tmp)       # 剩下的作为训练样本


    # 读取训练数据
    # 先把每个文件的训练数据同一到一起存入文件，再读出来
    traindatapath = savedata + '\\' + 'traindata.csv'
    with open(traindatapath, 'w', newline='', encoding='utf-8') as csvfile:
        csv_write = csv.writer(csvfile)
    for i in range(len(labelname)):
        for j in train_set[i]:
            csv_path = savefeature + '\\' + \
                labelname[i] + '\\' + os.listdir(savefeature + '\\' + labelname[i])[j]
            train = pd.read_csv(csv_path, encoding='utf-8', header=None)
            train = train.values.astype('float32')
            with open(traindatapath, 'a+', newline='', encoding='utf-8') as csvfile:
                csv_write = csv.writer(csvfile)
                csv_write.writerows(train)
    train = pd.read_csv(traindatapath, encoding='utf-8', header=None)
    train = train.values.astype('float32')
    train_X = train[:, 0:train.shape[1] - 2]
    train_Y = train[:, -2]


    # 训练KNN分类器
    N = 5
    neigh = KNeighborsClassifier(
        n_neighbors=5,
        algorithm='brute',
        metric='euclidean')
    # 使用我自己的KNN时，输入的特征矩阵列数代表样本数.使用库函数的KNN时正好相反。
    neigh.fit(train_X, train_Y)


    # 读取测试数据
    original_label = []
    predict_confidence = []
    predict_label = []
    for i in range(len(labelname)):                # 为了防止多次测试导致测试样本数据文件夹里的文件重复，因此先删除
        for j in test_set[i]:
            csv_path = savefeature + '\\' + \
                labelname[i] + '\\' + os.listdir(savefeature + '\\' + labelname[i])[j]
            test = pd.read_csv(csv_path, encoding='utf-8', header=None)
            test = test.values.astype('float32')
            test_X = test[:, 0:(test.shape[1] - 2)]
            test_Y = test[0, -2]
            predictions = neigh.predict(test_X)
            prediction, label_count = stats.mode(predictions)
            predict_label.append(prediction[0])  # 如果不用[0]，会导致存入数组对象而不是数
            original_label.append(test_Y)
            predict_confidence.append(label_count / test_X.shape[0])

    # 进行性能分析
    accuracy = accuracy_score(original_label, predict_label)  # 预测准确率
    visual.cm_plot(original_label, predict_label, pic=savepic + '\\' + 'cm_' + str(order))
    visual.confidece_plot(
        original_label,
        predict_label,
        predict_confidence,
        pic=savepic +
        '\\' +
        'confidence_' + str(order))

    return accuracy
