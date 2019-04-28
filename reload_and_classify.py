# -*- coding:utf8 -*-
# @TIME     : 2019/3/30 22:06
# @Author   : SuHao
# @File     : reload_and_classify.py

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


def reload_and_classify(feature_type,
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


    def gaussian_weight(neighbor, a=1, b=0, c=gaussian_bias):
        '''
        :param neighbor: 存放距离
        :return:
        '''
        gaussian_dist = gaussian(neighbor, a=a, b=b, c=c)
        return gaussian_dist


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
    test_set = []      # 每一类标签文件下的测试样本对应序号
    for i in sample_num:
        test_set.append(random.sample(range(0, i), int(i * test_rate)))       # 随机挑选样本作为测试样本
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
            train = pd.read_csv(csv_path, encoding='utf-8', header=None, sep=',')
            train = train.values.astype('float32')
            with open(traindatapath, 'a+', newline='', encoding='utf-8') as csvfile:
                csv_write = csv.writer(csvfile)
                csv_write.writerows(train)
    train = pd.read_csv(traindatapath, encoding='utf-8', header=None)
    train = train.values.astype('float32')
    train_X = train[:, 0:train.shape[1] - 2]
    if feature_select:
        train_X = train_X[:, feature_use]
    train_Y = train[:, -2]


    # 代价矩阵的转置
    train_num = np.bincount(train_Y.astype('int32'))
    train_num = np.log2(train_num)
    cost = np.tile(train_num, (len(train_num), 1))
    for diag in range(len(train_num)):
        cost[diag, diag] = 0


    #PCA
    if PCAornot:
        pass
        # for i in range(len(labelname)):
        #     # tmp = [0,1,2,3,5,6,9,10,11,12,13,14,15]
        #     label_i = np.where(train_Y == i)
        #     pca_input = train_X[label_i, :]   # 得到一个三维矩阵
        #     pca_input = pca_input[0,:,:]      # 转换成二维
        #     label_i_pca = visual.pca_plot(pca_model=None, x=pca_input, dim=3, pic=savepic+'\\'+str(order)+'_label_'+str(i))
        #     print(label_i_pca.explained_variance_ratio_)

        # label_0 = np.where(train_Y == 0)
        # x1 = train_X[label_0, :]
        # x1 = x1[0,:,:]
        # label_1 = np.where(train_Y == 1)
        # x2 = train_X[label_1, :]
        # x2 = x2[0,:,:]
        # visual.pca_2plot(x1=x1, x2=x2, dim=3, svd_solver='auto', pic=savepic+'\\'+str(order)+'_label_'+str(i))

    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # pca = PCA(n_components=70)
    # pca.fit(train_X)
    # train_X = pca.transform(train_X)
    # print(pca.explained_variance_ratio_)
    # ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], marker='o', c=train_Y)
    # ax.legend()
    # plt.show()
    #
    # lda = LinearDiscriminantAnalysis(n_components=3)
    # lda.fit(train_X, train_Y)
    # X_new = lda.fit_transform(train_X, train_Y)
    # plt.scatter(np.arange(0, len(X_new)), X_new, marker='o', c=train_Y)
    # plt.show()


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
    neigh.fit(train_X, train_Y)
    # neigh.fit(X_new, train_Y)


    # 读取测试数据
    original_label = []
    predict_confidence = []
    predict_label = []
    for i in range(len(labelname)):                # 为了防止多次测试导致测试样本数据文件夹里的文件重复，因此先删除
        for j in test_set[i]:
            csv_path = savefeature + '\\' + \
                labelname[i] + '\\' + os.listdir(savefeature + '\\' + labelname[i])[j]
            test = pd.read_csv(csv_path, encoding='utf-8', header=None, sep=',')
            test = test.values.astype('float32')
            test_X = test[:, 0:(test.shape[1] - 2)]
            if feature_select:
                test_X = test_X[:, feature_use]
            test_Y = test[0, -2]

            # test_X = pca.transform(test_X)                 # pca降维
            predictions = neigh.predict(test_X)

            # x_new = lda.transform(test_X)
            # predictions = neigh.predict(x_new)
            # predictions = lda.predict(test_X)

            if average:
                original_label.append(test_Y)
                prob = neigh.predict_proba(test_X)
                predic_tmp = predictions[0]
                prob_tmp = np.max(prob)

                if False:
                    ## 使用代价矩阵
                    all_cost = np.dot(cost, prob.T)
                    predic_tmp = np.argmin(all_cost)
                    prob_tmp = prob[0, predic_tmp]

                predict_label.append(predic_tmp)
                predict_confidence.append(prob_tmp)


            else:
                prediction, label_count = stats.mode(predictions)
                predict_label.append(prediction[0])  # 如果不用[0]，会导致存入数组对象而不是数
                original_label.append(test_Y)
                predict_confidence.append(label_count / test_X.shape[0])


    # 进行性能分析
    accuracy = accuracy_score(original_label, predict_label)  # 预测准确率
    visual.cm_plot(original_label, predict_label, labelname, accuracy*100, pic=savepic + '\\' + 'cm_' + str(order))
    visual.confidece_plot(
        original_label,
        predict_label,
        predict_confidence,
        pic=savepic +
        '\\' +
        'confidence_' + str(order))

    return accuracy
