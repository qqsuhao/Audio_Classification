# -*- coding:utf8 -*-
# @TIME     : 2019/4/27 12:12
# @Author   : SuHao
# @File     : load_and_frature_1.py
'''
frames 和 stft的列数不对齐的情况
'''
import numpy as np
import csv
import feature_extraction as fe
import os
import preprocessing
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
import visualization as visual
import stats as sts
import timbral_feature as timbral


def reload_and_feature(picall,
                       feature_type,
                       average,
                       nmel,
                       order_frft,
                       nmfcc,
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
    labelname = os.listdir(path)                            # 获取该数据集路径下的子文件名
    if not os.path.exists(savefeature):
        os.mkdir(savefeature)                               # 创建保存特征结果的文件
    for i in range(len(labelname)):
        if not os.path.exists(savefeature + '\\' + labelname[i]):
            os.mkdir(savefeature + '\\' + labelname[i])

    datafile = open(savepreprocess, encoding='utf-8')       # 读取预处理结果
    csv_reader = csv.reader(datafile)                       # 以这种方式读取文件得到的结果是一个迭代器

    feature_set = []                                        # 当使用统计量作为特征时，将所有样本的特征缓存入该变量以进行归一化
    for row in csv_reader:                                  # row中的元素是字符类型
        time_series = np.array(row[2:]).astype('float32')   # row的前两个元素分别是标签和对应文件次序
        #######################################################################
        frames = preprocessing.frame(time_series, frame_length, frame_overlap)                      # 分帧
        f, t, stft = fe.stft(time_series, pic=None, fs=downsample_rate, nperseg=frame_length,
                             noverlap=frame_overlap, nfft=8192, boundary=None, padded=False)
        if stft.shape[1] != frames.shape[1]:                                 # 防止stft的时间个数和帧的个数不一样
            dim = min(stft.shape[1], frames.shape[1])
            stft = stft[:, 0:dim]
            frames = frames[:, 0:dim]
        # Mel = lib.feature.melspectrogram(S=np.abs(stft), sr=downsample_rate, n_fft=2*(stft.shape[0]-1), n_mels=512)
        feature_list = []                       # 用于存放各种类型的特征，每个帧对应一个特征向量，其元素分别是每种类型的特征
        if picall:                              # 用于绘图控制
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
                feature9 = fe.mfccs(X=stft,fs=downsample_rate,
                                    # nfft=2*(stft.shape[0]-1),
                                    nfft=8192,
                                    n_mels=nmel,
                                    n_mfcc=nmfcc,
                                    pic=pic)
                feature_list.append(feature9)
            elif i == 10:
                feature10 = fe.rms(stft, pic=pic)
                feature_list.append(feature10)
            elif i == 11:
                feature11 = fe.stfrft(frames, p=order_frft[int(row[0])], pic=pic)
                feature_list.append(feature11)
            elif i == 12:
                tmp = fe.stfrft(frames, p=order_frft[int(row[0])])
                feature12 = fe.frft_MFCC(S=tmp, fs=downsample_rate, n_mfcc=nmfcc, n_mels=nmel, pic=pic)
                feature_list.append(feature12)
            elif i == 13:
                feature13, feature13_ = fe.fundalmental_freq(frames=frames, fs=downsample_rate, pic=pic)
                feature_list.append(feature13)
            elif i == 14:
                feature14 = fe.chroma_stft(S=stft, n_chroma=12, A440=440.0, ctroct=5.0, octwidth=2, base_c=True, norm=2)
                feature_list.append(feature14)
            elif i == 15:
                feature15 = fe.log_attack_time(x=time_series, lower_ratio=0.02, upper_ratio=0.99, fs=downsample_rate, n=frames.shape[1])
                feature_list.append(feature15)
            elif i == 16:
                feature16 = fe.temoporal_centroid(S=stft, hop_length=frame_overlap, fs=downsample_rate)
                feature_list.append(feature16)
            elif i == 17:
                # harm_freq, harm_mag = fe.harmonics(nfft=8192, nht=0.15, f=f, S=stft, fs=downsample_rate, fmin=50, fmax=500, threshold=0.2)
                # hsc = fe.harmonic_spectral_centroid(harm_freq, harm_mag)
                # hsd = fe.harmonic_spectral_deviation(harm_mag)
                # hss = fe.harmonic_spectral_spread(hsc, harm_freq, harm_mag)
                # hsv = fe.harmonic_spectral_variation(harm_mag)
                # feature17 = np.concatenate([hsc, hsd, hss, hsv], axis=0)
                # feature_list.append(feature17)
                harm_freq, harm_mag = timbral.harmonics(frames=frames, fs=downsample_rate, S=stft, f=f, nfft=8192, fmin=50, fmax=500,  nht=0.15)
                hsc = timbral.harmonic_spectral_centroid(harm_freq, harm_mag)
                hsd = timbral.harmonic_spectral_deviation(harm_mag)
                hss = timbral.harmonic_spectral_spread(hsc, harm_freq, harm_mag)
                hsv = timbral.harmonic_spectral_variation(harm_mag)
                feature17 = np.concatenate([hsc, hsd, hss, hsv], axis=0)
                feature_list.append(feature17)
            elif i == 18:
                feature18 = fe.pitches_mag_CDSV(f=f, S=stft, fs=downsample_rate, fmin=50, fmax=downsample_rate/2, threshold=0.2)
                feature_list.append(feature18)
            elif i == 19:
                feature19 = fe.delta_features(feature9, order=1)
                feature_list.append(feature19)
            elif i == 20:
                feature20 = fe.delta_features(feature9, order=2)
                feature_list.append(feature20)
            elif i == 21:
                harm_freq, harm_mag = timbral.harmonics(frames=frames, fs=downsample_rate, S=stft, f=f, nfft=8192, fmin=50, fmax=500,  nht=0.15)
                hsc = timbral.harmonic_spectral_centroid(harm_freq, harm_mag)
                hsd = timbral.harmonic_spectral_deviation(harm_mag)
                hss = timbral.harmonic_spectral_spread(hsc, harm_freq, harm_mag)
                hsv = timbral.harmonic_spectral_variation(harm_mag)
                feature21 = np.concatenate([hsc, hsd, hss, hsv], axis=0)
                feature_list.append(feature21)

        features = np.concatenate([j for j in feature_list], axis=0)    # 我很欣赏这一句代码，将各种特征拼在一起
        long = list(range(features.shape[1]))                           # 删除含有nan的帧
        for t in long[::-1]:
            if np.isnan(features[:, t]).any():
                features = np.delete(features, t, 1)
        if average:                  # 使用统计量作为特征
            mean = np.mean(features, axis=1).reshape(1, features.shape[0])  # 原来的特征向量是列向量，这里转成行向量
            var = np.var(features, axis=1).reshape(1, features.shape[0])
            # std = np.std(features, axis=1).reshape(1, features.shape[0])
            # ske = np.zeros((1, features.shape[0]))
            # kur = np.zeros((1, features.shape[0]))
            # for n in range(features.shape[0]):
            #     ske[0, i] = sts.skewness(features[i, :])
            #     kur[0, i] = sts.kurtosis(features[i, :])
            features = np.concatenate([mean, var, np.array([int(row[0]), int(row[1])]).reshape(1, 2)], axis=1) # 使用统计平均代替每个帧的特征
            feature_set.append(features)
        else:
            scale = StandardScaler().fit(features)
            features = scale.transform(features)            # 进行归一化
            csv_path = savefeature + '\\' + labelname[int(row[0])] + '\\' + row[0] + '_' + row[1] + '.csv'
            with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                buffer = np.concatenate([features.T, int(row[0]) *
                                         np.ones((features.shape[1], 1)), int(row[1]) *
                                         np.ones((features.shape[1], 1))], axis=1)
                csv_writer.writerows(buffer)
        print('featuring:', row[0], row[1])
    datafile.close()                                                    # 关闭文件，避免不必要的错误
    if average:                                                         # 使用统计量作为特征
        features = np.concatenate([k for k in feature_set], axis=0)     # 我很欣赏这一句代码    行表示样本数，列表示特征数
        tmp = features[:, -2:]                                          # 防止归一化的时候把标签也归一化
        features = features[:, 0:-2]
        scale = StandardScaler().fit(features)
        features = scale.transform(features)                            # 进行归一化
        features = np.concatenate([features, tmp], axis=1)              # 把之前分开的特征和标签拼在一起
        for k in range(features.shape[0]):
            csv_path = savefeature + '\\' + labelname[int(features[k, -2])] + \
                       '\\' + str(int(features[k, -2])) + '_' + str(int(features[k, -1])) + '.csv'
            with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)            # 每个音频文件只有一个特征向量，并存入一个csv文件
                csv_writer.writerow(features[k, :])         # 注意这里写入的是一行，要用writerow
