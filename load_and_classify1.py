# -*- coding:utf8 -*-
# @TIME     : 2019/5/7 7:59
# @Author   : SuHao
# @File     : load_and_classify1.py
'''
使用库函数实现的留出法和交叉验证，只适用于每个文件对应一个特征。
'''

import math
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
from scipy import stats
import visualization as visual
import random
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from minepy import MINE


def gaussian(dist, a=1, b=0, c=0.8):
    '''
    :param dist: 距离
    :param a: 加权系数
    :param b: 均值
    :param c: 标准差
    :return:
    '''
    if dist.shape[0] == 1 and dist.shape[1] == 1:
        return a * math.e ** (-(dist - b) ** 2 / (2 * c ** 2))
    d = (dist - np.mean(dist))/np.std(dist)    #高斯方法采用正态归一化
    return a * math.e ** (-(d - b) ** 2 / (2 * c ** 2))

def reload(feature_reduce,
            feature_type,
            feature_select,
            order,
            PCAornot,
            average,
            neighbors,
            metric,
            weight_type,
            gaussian_bias,
            dist_p,
            saveprojectpath,
            savedata,
            savepic,
            savetestdata,
            savepreprocess,
            savefeature,
            path,
            downsample_rate,
            frame_time,
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
    feature_use = []
    feature_length = [frame_length // 2 + 1,        # 0.stft
                      1,                              # 1.zero_crossing_rate
                      1,                              # 2.energy
                      1,                              # 3.entropy_of_energy
                      2,                              # 4.spectral_centroid_spread
                      1,                              # 5.spectral_entropy
                      1,                              # 6.spectral_flux
                      1,                              # 7.spectral_rolloff
                      1,                              # 8.bandwidth
                      13,                              # 9.mfccs
                      1,                              # 10.rms
                      frame_length // 2,           # 11.stfrft
                      13,                           # 12.frft_mfcc
                      1,                              # 13.Harmonics
                      12,                              # 14.chroma_stft
                      1,                                # 15.log attack time
                      1,                                # 16.temporal centroid
                      4,                               # 17.harmonic spectral CDSV
                      4,                            # 18.pitches mag CDSV(指定频率范围的音色特征)
                      13,                            # 19.1-order delta of mfccs
                      13]                            # 20.2-order delta of mfccs

    # 记录每种特征的向量长度
    if feature_select:
        feature_length = [feature_length[i] for i in feature_type]  # 把每种特征对应的长度挑出来 feature_length和feature_type一一对应
        for i in feature_select:
            a = feature_type.index(i)
            feature_use += [sum(feature_length[0:a])+j for j in list(range(0, feature_length[a]))]  # 这句代码写的真好，佩服自己，哈哈哈哈
        feature_use += [sum(feature_length) + i for i in feature_use]


    if not os.path.exists(savepic):
        os.mkdir(savepic)    # 创建储存图片的文件夹
    sample_num = []
    labelname = os.listdir(savefeature)
    for i in range(len(labelname)):
        sample_num.append(len(os.listdir(savefeature + '\\' + labelname[i])))


    # 读取训练数据
    # 先把每个文件的训练数据同一到一起存入文件，再读出来
    traindatapath = savedata + '\\' + 'traindata.csv'
    with open(traindatapath, 'w', newline='', encoding='utf-8') as csvfile:
        csv_write = csv.writer(csvfile)
    for i in range(len(labelname)):
        for j in range(sample_num[i]):
            csv_path = savefeature + '\\' + \
                       labelname[i] + '\\' + os.listdir(savefeature + '\\' + labelname[i])[j]
            train = pd.read_csv(csv_path, encoding='utf-8', header=None, sep=',')
            train = train.values.astype('float32')
            with open(traindatapath, 'a+', newline='', encoding='utf-8') as csvfile:
                csv_write = csv.writer(csvfile)
                csv_write.writerows(train)
    train = pd.read_csv(traindatapath, encoding='utf-8', header=None)
    train = train.values.astype('float32')
    X = train[:, 0:train.shape[1] - 2]
    if feature_select:
        X = X[:, feature_use]
    Y = train[:, -2]


    # 由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
    def mic(x, y):
        m = MINE()
        m.compute_score(x, y)
        # return (m.mic(), 0.9)
        return m.mic()

    # 选择K个最好的特征，返回特征选择后的数据
    # X = SelectKBest(lambda A, B: tuple(np.array(list(map(lambda a: mic(a, B), A.T))).T), k=50).fit_transform(X, Y)
    if feature_reduce > 0:
        model = SelectKBest(lambda A, B: np.array(list(map(lambda a: mic(a, B), A.T))), k=feature_reduce)
        X = model.fit_transform(X, Y)
        print(model.get_support(True))
    return X, Y


def classify(X,Y,valid_k,
            feature_type,
            feature_select,
            order,
            PCAornot,
            average,
            neighbors,
            metric,
            weight_type,
            gaussian_bias,
            dist_p,
            saveprojectpath,
            savedata,
            savepic,
            savetestdata,
            savepreprocess,
            savefeature,
            path,
            downsample_rate,
            frame_time,
            frame_length,
            frame_overlap,
            test_rate):
    labelname = os.listdir(savefeature)

    def gaussian_weight(neighbor, a=1, b=0, c=gaussian_bias):
        '''
        :param neighbor: 存放距离
        :return:
        '''
        gaussian_dist = gaussian(neighbor, a=a, b=b, c=c)
        return gaussian_dist


    # 训练KNN分类器
    N = neighbors
    if weight_type == 'gaussian':
        neigh = KNeighborsClassifier(
            n_neighbors=N,
            algorithm='brute',
            metric=metric,
            weights=gaussian_weight,
            p=dist_p)
    elif weight_type == 'distance':
        neigh = KNeighborsClassifier(
            n_neighbors=N,
            algorithm='brute',
            metric=metric,
            weights='distance',
            p=dist_p)
    elif weight_type == 'uniform':
        neigh = KNeighborsClassifier(
            n_neighbors=N,
            algorithm='brute',
            metric=metric,
            weights='uniform',
            p=dist_p)
    # 使用我自己的KNN时，输入的特征矩阵列数代表样本数.使用库函数的KNN时正好相反。


    if valid_k == 0:
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=test_rate, stratify=Y)
        neigh.fit(train_X, train_Y)
        predictions = neigh.predict(test_X)
        predict_label = predictions
        original_label = test_Y
        predict_confidence = neigh.predict_proba(test_X)
        accuracy = accuracy_score(original_label, predict_label)  # 预测准确率
        visual.cm_plot(original_label, predict_label, labelname, accuracy*100, pic=savepic + '\\' + 'cm_' + str(order))
    else:
        accuracy_10 = []
        skf = StratifiedKFold(n_splits=valid_k, shuffle=True)
        i = 0
        for train_index, test_index in skf.split(X, Y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            train_X, test_X = X[train_index], X[test_index]
            train_Y, test_Y = Y[train_index], Y[test_index]
            neigh.fit(train_X, train_Y)
            predictions = neigh.predict(test_X)
            predict_label = predictions
            original_label = test_Y
            predict_confidence = neigh.predict_proba(test_X)
            # 进行性能分析
            i += 1
            accuracy = accuracy_score(original_label, predict_label)  # 预测准确率
            visual.cm_plot(original_label, predict_label, labelname, accuracy*100, pic=savepic + '\\' + 'cm_' + str(order)+str(i))
            accuracy_10.append(accuracy)



        # lda = LinearDiscriminantAnalysis(n_components=3)
        # accuracy_10 = []
        # skf = StratifiedKFold(n_splits=valid_k, shuffle=True)
        # i = 0
        # for train_index, test_index in skf.split(X, Y):
        #     # print("TRAIN:", train_index, "TEST:", test_index)
        #     train_X, test_X = X[train_index], X[test_index]
        #     train_Y, test_Y = Y[train_index], Y[test_index]
        #     lda.fit(train_X, train_Y)
        #     train_X_new = lda.fit_transform(train_X, train_Y)
        #     # test_X_new = lda.fit_transform(test_X, test_Y)
        #     plt.figure()
        #     plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=train_Y)
        #     plt.plot(lda.explained_variance_ratio_)
        #     plt.show()
        #     predictions = lda.predict(test_X)
        #     predict_label = predictions
        #     original_label = test_Y
        #     #
        #     # KNN
        #     # neigh.fit(train_X_new, train_Y)
        #     # predictions = neigh.predict(test_X_new)
        #     # predict_label = predictions
        #     # original_label = test_Y
        #     # 进行性能分析
        #     i += 1
        #     accuracy = accuracy_score(original_label, predict_label)  # 预测准确率
        #     visual.cm_plot(original_label, predict_label, labelname, accuracy*100, pic=savepic + '\\' + 'cm_' + str(order)+str(i))
        #     accuracy_10.append(accuracy)


    if valid_k > 0:
        return sum(accuracy_10)/valid_k
    else:
        return accuracy
