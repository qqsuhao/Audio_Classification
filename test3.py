# -*- coding:utf8 -*-
# @TIME     : 2019/3/30 18:21
# @Author   : SuHao
# @File     : test3.py


'''
主要以男女两类人声数据进行程序测试
主要测试内容：
    1.频谱图是否正确。能否从频谱图上看出两类数据的差异。
    2.特征提取代码能否正确运行。
    3.使用KNN能否实现高正确率分类。
'''

import numpy as np
import librosa
import os
import json
import matplotlib.pyplot as plt
import preprocessing
import time
import visualization as visual
import csv

if not os.path.exists('..\\仿真结果'):
    os.mkdir('..\\仿真结果')
saveprojectpath = '..\\仿真结果\\boy_and_girl_without_pre_amphasis_hilbert'
savedata = saveprojectpath + '\\data'
savepic = saveprojectpath + '\\pic'
if not os.path.exists(saveprojectpath):
    os.mkdir(saveprojectpath)
if not os.path.exists(savedata):
    os.mkdir(savedata)
if not os.path.exists(savepic):
    os.mkdir(savepic)


# 写入数据
with open(savedata+'\\'+'preprocessing_result.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    # writer.writerow(["index","a_name","b_name"])  # 写入列名
    # writer.writerows([[0, 1, 3], [1, 2, 3], [2, 3, 4]]) # 写入多行用writerows


path = '..\\boy_and_girl'
labelname = os.listdir(path)   # 获取该路径下的子文件名
audio_data_set = dict()


downsample_rate = 8000
frame_length = int(0.02 * downsample_rate)  # 20ms
frame_overlap = frame_length // 2


sample_num = []           #统计每个标签的样本数
for j in range(len(labelname)):
    subpath = path + '\\' + labelname[j]
    subfilename = os.listdir(subpath)    #查看音频文件目录
    sample_num.append(len(subfilename))   # 统计文件数量，也可以认为是统计样本数量
    for i in range(len(subfilename)):
        audio_data, sample_rate = librosa.load(subpath + '\\' + subfilename[i], sr=None, mono=True, res_type='kaiser_best') # 读取文件
        # pre_amphasis = preprocessing.pre_emphasis(audio_data, 0.97, pic=savepic + '\\' + 'pre_amphasis_'+str(j)+'_'+str(i))
        pre_amphasis = audio_data
        avoid_overlap = preprocessing.avoid_overlap(pre_amphasis,
                                                    N=10,
                                                    f=4000,
                                                    fs=sample_rate,
                                                    plot=False)
        downsample = preprocessing.downsample(
            avoid_overlap, sample_rate, downsample_rate)
        silence_remove = preprocessing.silence_remove(
            downsample,
            limit=np.max(downsample) / 20,
            option='hilbert',
            # pic=savepic + '\\' + 'silence_remove_hilbert_' + str(j)+'_'+str(i)
            pic=None)
        # _, _, _ = visual.stft_specgram(silence_remove, picname=savepic + '\\' + 'stft_'+str(j)+'_'+str(i), fs=downsample_rate,
        #                                nperseg=512, noverlap=128, nfft=1024)
        buffer = [j] + list(silence_remove)

        # 写入数据
        with open(savedata+'\\'+'preprocessing_result.csv', 'a+', newline='') as csvfile:
            csv_write = csv.writer(csvfile)
            csv_write.writerow(buffer)

        print(j,i)
