# -*- coding:utf8 -*-
# @TIME     : 2019/3/30 20:50
# @Author   : SuHao
# @File     : load_and_preprocess.py

import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
import preprocessing
import time
import visualization as visual
import csv

'''
对于不同的数据集，进行测试时，只需要更改saveprojectpath 和 path
(不包括预处理等其他内容的详细参数设置)
'''

saveprojectpath = '..\\仿真结果\\boy_and_girl_without_pre_amphasis_hilbert'
savedata = saveprojectpath + '\\data'
savepic = saveprojectpath + '\\pic'
savetestdata = savedata + '\\' + 'test_data'
savepreprocess = savedata + '\\' + 'preprocessing_result.csv'
savetrainfeature = savedata + '\\' + 'feature_result_train.csv'
savetestfeature = savedata + '\\' + 'test_data' + '\\' + 'feature_result_test.csv'
path = '..\\boy_and_girl'  # 数据集路径
downsample_rate = 8000
frame_length = int(0.02 * downsample_rate)  # 20ms
frame_overlap = frame_length // 2


if not os.path.exists('..\\仿真结果'):
    os.mkdir('..\\仿真结果')
if not os.path.exists(saveprojectpath):
    os.mkdir(saveprojectpath)
if not os.path.exists(savedata):
    os.mkdir(savedata)
if not os.path.exists(savepic):
    os.mkdir(savepic)
if not os.path.exists(savetestdata):
    os.mkdir(savetestdata)


# 写入数据
with open(savepreprocess, 'w', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
# 读取音频文件目录
labelname = os.listdir(path)   # 获取该路径下的子文件名
for j in range(len(labelname)):
    subpath = path + '\\' + labelname[j]
    subfilename = os.listdir(subpath)  # 查看音频文件目录
    for i in range(len(subfilename)):
        audio_data, sample_rate = librosa.load(
            subpath + '\\' + subfilename[i], sr=None, mono=True, res_type='kaiser_best')  # 读取文件
        ##############################################################################################################


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


        ###################################################################################################
        buffer = [j] + [i] + list(silence_remove)     # j表示标签， i表示同一标签下的音频文件序号
        with open(savepreprocess, 'a+', newline='', encoding='utf-8') as csvfile:
            csv_write = csv.writer(csvfile)
            csv_write.writerow(buffer)

        print(j, i)
