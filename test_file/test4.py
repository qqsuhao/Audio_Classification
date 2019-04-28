# -*- coding:utf8 -*-
# @TIME     : 2019/3/30 21:00
# @Author   : SuHao
# @File     : test4.py


import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
import preprocessing
import time
import visualization as visual
import csv
import pandas as pd
import feature_extraction as fea


'''
二分类人生数据，紧接test3的程序
计算特征并存储。
'''

saveprojectpath = '..\\仿真结果\\boy_and_girl_without_pre_amphasis_hilbert'
savedata = saveprojectpath + '\\data'
savepic = saveprojectpath + '\\pic'
path = '..\\boy_and_girl'  # 数据集路径
downsample_rate = 8000
frame_length = int(0.02 * downsample_rate)  # 20ms
frame_overlap = frame_length // 2


with open(savedata + '\\' + 'feature_result.csv', 'w', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)


datafile = open(savedata + '\\' + 'preprocessing_result.csv')
csv_reader = csv.reader(datafile)
for row in csv_reader:
    time_series = np.array(row[1:]).astype('float32')
    # stft的每一列一个帧对应的特征
    _, _, stft = fea.stft(time_series, fs=downsample_rate,
                          nperseg=512, noverlap=128, nfft=512)
    stft = np.abs(stft)
    for i in range(stft.shape[1]):
        with open(savedata + '\\' + 'feature_result.csv', 'a+', newline='', encoding='utf-8') as csvfile:
            csv_write = csv.writer(csvfile)
            buffer = list(stft[:, i]) + [row[0]]    # 把特征对应的标签加进来
            csv_write.writerow(buffer)
            print(row[0], i)
