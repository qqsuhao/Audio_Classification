# -*- coding:utf8 -*-
# @TIME     : 2019/3/27 9:24
# @Author   : SuHao
# @File     : test1.py
import numpy as np
import librosa
import os
import json
import matplotlib.pyplot as plt
import preprocessing
import time

path = '..\\数据集2\\post2012'
labelname = os.listdir(path)   # 获取该路径下的子文件名
audio_data_set = dict()


sample_num = []
for i in range(1):
    subpath = path + '\\' + labelname[i]
    subfilename = os.listdir(subpath)
    sample_num.append(len(subfilename))   # 统计文件数量，也可以认为是统计样本数量
    list = []
    audio_data, sample_rate = librosa.load(
        subpath + '\\' + subfilename[0], sr=None, mono=True, res_type='kaiser_best')
    list.append(audio_data.tolist())
    audio_data_set[labelname[i]] = list

for i in range(1):
    data = np.array(audio_data_set[labelname[i]][0])
    pre_amphasis = preprocessing.pre_emphasis(data, 0.97)
    avoid_overlap = preprocessing.avoid_overlap(pre_amphasis,
                                N=10,
                                f=11000,
                                fs=44100,
                                plot=True)
    downsample = preprocessing.downsample(avoid_overlap, 44100, 22050)
    start = time.perf_counter()
    silence_remove = preprocessing.silence_remove(
        downsample,
        limit=0.001,
        option=filter,
        pic=True,
        N=10,
        f=100,
        fs=22050,
        plot=True)
    end = time.perf_counter()
    print(end-start)

    start = time.perf_counter()
    silence_remove2 = preprocessing.silence_remove(
            downsample,
            limit=None,
            option='SVM',
            pic=True,
            fs=22050,
            st_win=np.array(0.02).astype('float32'),
            st_step=np.array(0.01).astype('float32'),
            smoothWindow=0.5,
            weight=0.2,
            plot=True)
    end = time.perf_counter()
    print(end-start)

    start = time.perf_counter()
    silence_remove3 = preprocessing.silence_remove(
            downsample,
            limit=np.max(downsample)/20,
            option='hilbert',
            pic=True)
    end = time.perf_counter()
    print(end-start)
