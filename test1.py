# -*- coding:utf8 -*-
# @TIME     : 2019/3/27 9:24
# @Author   : SuHao
# @File     : test1.py
import numpy as np
import librosa
# import pyAudioAnalysis as paa
# from pyAudioAnalysis import audioBasicIO as io
import os
import json
import matplotlib.pyplot as plt
import preprocessing

path = '..\\数据集2\\pre2012'
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
    downsample = preprocessing.downsample(avoid_overlap, 44100, 22000)
    silence_remove = preprocessing.silence_remove(
        downsample,
        limit=0.0005,
        option=filter,
        pic=True,
        N=10,
        f=600,
        fs=22000,
        plot=True)
    plt.figure(figsize=(8,8),dpi=300)
    ax1 = plt.subplot2grid((4,4),(0,0), colspan=4, rowspan=2)
    ax1.plot(downsample, 'r')
    ax1.set_title('original signal')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('value')
    ax2 = plt.subplot2grid((4,4),(2,0), colspan=4, rowspan=2)
    ax2.plot(silence_remove, 'b')
    ax2.set_title('silence_remove_filter')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('value')
    plt.savefig('.\\picture\\silence_remove_filter.jpg')
    plt.show()
    silence_remove2 = preprocessing.silence_remove(
            downsample,
            limit=0.0005,
            option='hilbert',
            pic=True)
    plt.figure(figsize=(8,8),dpi=300)
    ax1 = plt.subplot2grid((4,4),(0,0), colspan=4, rowspan=2)
    ax1.plot(downsample, 'r')
    ax1.set_title('original signal')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('value')
    ax2 = plt.subplot2grid((4,4),(2,0), colspan=4, rowspan=2)
    ax2.plot(silence_remove2, 'b')
    ax2.set_title('silence_remove_hilbert')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('value')
    plt.savefig('.\\picture\\silence_remove_hilbert.jpg')
    plt.show()
