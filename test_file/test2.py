# -*- coding:utf8 -*-
# @TIME     : 2019/3/29 9:27
# @Author   : SuHao
# @File     : test2.py

import numpy as np
import librosa
import os
import json
import matplotlib.pyplot as plt
import preprocessing
import time
import visualization as visual

'''
主要测试绘制频谱图的功能
观察一些测试样本的频谱图的区别。
'''

path = '..\\数据集2\\post2012'
labelname = os.listdir(path)   # 获取该路径下的子文件名
audio_data_set = dict()

downsample_rate = 22050
frame_length = int(0.02 * downsample_rate)  # 20ms
frame_overlap = frame_length // 2

sample_num = []           #统计每个标签的样本数
for j in range(len(labelname)):
    subpath = path + '\\' + labelname[j]
    subfilename = os.listdir(subpath)    #查看音频文件目录
    sample_num.append(len(subfilename))   # 统计文件数量，也可以认为是统计样本数量

    for i in range(0,6,1):
        audio_data, sample_rate = librosa.load(subpath + '\\' + subfilename[i], sr=None, mono=True, res_type='kaiser_best') # 读取文件
        # pre_amphasis = preprocessing.pre_emphasis(audio_data, 0.97, pic='pre_amphasis_'+str(j)+'_'+str(i))
        pre_amphasis = audio_data * 10
        avoid_overlap = preprocessing.avoid_overlap(pre_amphasis,
                                                    N=10,
                                                    f=11000,
                                                    fs=sample_rate,
                                                    plot=None)
        downsample = preprocessing.downsample(
            avoid_overlap, sample_rate, downsample_rate)
        silence_remove = preprocessing.silence_remove(
            downsample,
            limit=None,
            option='SVM',
            pic='silence_remove_SVM_'+str(j)+'_'+str(i),
            fs=downsample_rate,
            st_win=np.array(0.02).astype('float32'),
            st_step=np.array(0.01).astype('float32'),
            smoothWindow=0.1,
            weight=0.1,
            plot=False)
        silence_remove = downsample
        _, _, _ = visual.stft_specgram(silence_remove, picname='stft_'+str(j)+'_'+str(i), fs=downsample_rate,
                                       nperseg=2048, noverlap=256, nfft=2048)
        print(j,i)
