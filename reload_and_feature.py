# -*- coding:utf8 -*-
# @TIME     : 2019/3/30 21:58
# @Author   : SuHao
# @File     : reload_and_feature.py


import numpy as np
import csv
import feature_extraction as fe
import os
import preprocessing
from sklearn.preprocessing import minmax_scale


def reload_and_feature(picall,
                       feature_type,
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
    fe.stft,                           # 0
    fe.zero_crossing_rate,             # 1
    fe.energy,                         # 2
    fe.entropy_of_energy,              # 3
    fe.spectral_centroid_spread,       # 4
    fe.spectral_entropy,               # 5
    fe.spectral_flux,                  # 6
    fe.spectral_rolloff,               # 7
    fe.bandwidth,                      # 8
    fe.mfccs,                          # 9
    fe.rms                             # 10
    fe.stfrft                          # 11
    fe.frft_mfcc                       # 12

    '''
    labelname = os.listdir(path)       # 获取该数据集路径下的子文件名
    if not os.path.exists(savefeature):
        os.mkdir(savefeature)          # 创建保存特征结果的文件
    for i in range(len(labelname)):
        if not os.path.exists(savefeature + '\\' + labelname[i]):
            os.mkdir(savefeature + '\\' + labelname[i])
    datafile = open(savepreprocess, encoding='utf-8')      # 读取预处理结果
    csv_reader = csv.reader(datafile)
    for row in csv_reader:            # row中的元素是字符类型
        time_series = np.array(row[2:]).astype('float32')
        #######################################################################
        frames = preprocessing.frame(time_series, frame_length, frame_overlap)
        f, t, stft = fe.stft(time_series, pic=savepic + '\\' + row[0] + '_' + row[1], fs=downsample_rate, nperseg=frame_length,
                             noverlap=frame_overlap, nfft=frame_length, padded=True, boundary=None)
        if stft.shape[1] != frames.shape[1]:                   #防止stft的时间个数和帧的个数不一样
            dim = min(stft.shape[1], frames.shape[1])
            stft = stft[:, 0:dim]
            frames = frames[:, 0:dim]
        # Mel = lib.feature.melspectrogram(S=np.abs(stft), sr=downsample_rate, n_fft=2*(stft.shape[0]-1), n_mels=512)
        feature_list = []
        # 用于绘图控制
        if picall:
            pic = savepic + '\\' + row[0] + '_' + row[1]
        else:
            pic = None
        for i in feature_type:
            if i == 0:
                feature0 = np.abs(stft)
                feature_list.append(feature0)
            elif i == 1:
                feature1 = fe.zero_crossing_rate(frames, pic=pic)
                feature_list.append(feature1)
            elif i == 2:
                feature2 = fe.energy(frames, pic=pic)
                feature_list.append(feature2)
            elif i == 3:
                feature3 = fe.entropy_of_energy(frames, pic=pic)
                feature_list.append(feature3)
            elif i == 4:
                feature4, feature41 = fe.spectral_centroid_spread(stft, downsample_rate, pic=pic)
                feature_list.append(feature4)
                feature_list.append(feature41)
            elif i == 5:
                feature5 = fe.spectral_entropy(stft, pic=pic)
                feature_list.append(feature5)
            elif i == 6:
                feature6 = fe.spectral_flux(stft, pic=pic)
                feature_list.append(feature6)
            elif i == 7:
                feature7 = fe.spectral_rolloff(stft, 0.85, downsample_rate, pic=pic)
                feature_list.append(feature7)
            elif i == 8:
                feature8 = fe.bandwidth(stft, f, pic=pic)
                feature_list.append(feature8)
            elif i == 9:
                feature9 = fe.mfccs(X=stft,
                                    fs=downsample_rate,
                                    nfft=2*(stft.shape[0]-1),
                                    n_mels=512,
                                    n_mfcc=13,
                                    pic=pic)
                feature_list.append(feature9)
            elif i == 10:
                feature10 = fe.rms(stft, pic=pic)
                feature_list.append(feature10)
            elif i == 11:
                feature11 = fe.stfrft(frames, p=0.95, pic=pic)
                feature_list.append(feature11)
            elif i == 12:
                tmp = fe.stfrft(frames, p=0.95)
                feature12 = fe.frft_MFCC(S=tmp, fs=downsample_rate, n_mfcc=13, n_mels=512, pic=pic)
                feature_list.append(feature12)
            elif i == 13:
                feature13, feature13_ = fe.fundalmental_freq(frames=frames, fs=downsample_rate, pic=pic)
                feature_list.append(feature13)

        features = np.concatenate(
            [j for j in feature_list], axis=0)    # 我很欣赏这一句代码
        # mean = np.mean(features, axis=1).reshape(features.shape[0],1)
        # var = np.var(features, axis=1).reshape(features.shape[0],1)
        # features = np.concatenate([mean, var], axis=1)            # 使用统计平均代替每个帧的特征
        # features = minmax_scale(features, feature_range=(0, 1), axis=0, copy=True)
        #######################################################################
        csv_path = savefeature + '\\' + \
            labelname[int(row[0])] + '\\' + row[0] + '_' + row[1] + '.csv'
        with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            buffer = np.concatenate([features.T, int(row[0]) *
                                     np.ones((features.shape[1], 1)), int(row[1]) *
                                     np.ones((features.shape[1], 1))], axis=1)
            csv_writer.writerows(buffer)

        print('featuring:', row[0], row[1])
