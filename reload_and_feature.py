# -*- coding:utf8 -*-
# @TIME     : 2019/3/30 21:58
# @Author   : SuHao
# @File     : reload_and_feature.py


import numpy as np
import csv
import feature_extraction as fea



saveprojectpath = '..\\仿真结果\\boy_and_girl_without_pre_amphasis_hilbert'
savedata = saveprojectpath + '\\data'
savepic = saveprojectpath + '\\pic'
savepreprocess = savedata + '\\' + 'preprocessing_result.csv'
savefeature = savedata + '\\' + 'feature_result.csv'
path = '..\\boy_and_girl'  # 数据集路径
downsample_rate = 8000
frame_length = int(0.02 * downsample_rate)  # 20ms
frame_overlap = frame_length // 2


with open(savefeature, 'w', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
datafile = open(savepreprocess, encoding='utf-8')
csv_reader = csv.reader(datafile)
for row in csv_reader:
    time_series = np.array(row[1:]).astype('float32')
    ###############################################################################################


    # stft的每一列一个帧对应的特征
    _, _, features = fea.stft(time_series, fs=downsample_rate,
                          nperseg=512, noverlap=128, nfft=512)
    features = np.abs(features)




    ################################################################################################
    for i in range(features.shape[1]):
        with open(savefeature, 'a+', newline='', encoding='utf-8') as csvfile:
            csv_write = csv.writer(csvfile)
            buffer = list(features[:, i]) + [row[0]]    # 把特征对应的标签加进来
            csv_write.writerow(buffer)
            print(row[0], i)
