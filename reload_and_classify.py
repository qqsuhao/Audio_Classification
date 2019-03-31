# -*- coding:utf8 -*-
# @TIME     : 2019/3/30 22:06
# @Author   : SuHao
# @File     : reload_and_classify.py

import csv
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
import Su_KNN as KNN


saveprojectpath = '..\\仿真结果\\boy_and_girl_without_pre_amphasis_hilbert'
savedata = saveprojectpath + '\\data'
savepic = saveprojectpath + '\\pic'
savepreprocess = savedata + '\\' + 'preprocessing_result.csv'
savefeature = savedata + '\\' + 'feature_result.csv'
path = '..\\boy_and_girl'  # 数据集路径
downsample_rate = 8000
frame_length = int(0.02 * downsample_rate)  # 20ms
frame_overlap = frame_length // 2


features = pd.read_csv(savefeature, encoding='utf-8', header=None)
features = features.values.astype('float32')

train = features[,:]

'''
使用我自己的KNN时，输入的特征矩阵列数代表样本数
使用库函数的KNN时正好相反。
'''
N = 5
start1 = time.perf_counter()
neigh = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='brute',
    metric='euclidean')
neigh.fit(train_X, train_Y)
Pe1 = neigh.score(test_X[0:N, :], test_Y[0:N])
# dist1, result1 = neigh.kneighbors(X=test_X[1:2, :], n_neighbors=5, return_distance=True)
# print(dist1)
end1 = time.perf_counter()
print(end1 - start1, Pe1)


params = {'dist_form': None,
          'test_X': test_X.T[:, 0:N],
          'test_Y': test_Y[0:N],
          'k': 5,
          'kind': 'heapsort'}
start = time.perf_counter()
test = KNN.classifier('force', None)
test.train(train_X.T, train_Y, None)
pe = test.test(**params)
# predict_result, predict_prob = test.predict(dist_form=None, k=5, test_X=test_X[:, 0:N], kind='heapsort')
end = time.perf_counter()
print(end - start, pe)
