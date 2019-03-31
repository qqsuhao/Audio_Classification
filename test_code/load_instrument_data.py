# -*- coding:utf8 -*-
# @TIME     : 2019/3/25 20:54
# @Author   : SuHao
# @File     : load_instrument_data.py

import numpy as np
import librosa
# import pyAudioAnalysis as paa
# from pyAudioAnalysis import audioBasicIO as io
import os
import json
import matplotlib.pyplot as plt

'''
文件夹“数据集2”下有多个子文件，其文件名为每种乐器的名字，因此也是标签。
每个乐器文件夹里有多个音频文件因此每个乐器文件夹需要用一个二维数组来存放数据。
依次读取每个乐器文件夹，然后每个文件夹产生的二维数组存入一个列表，顺序对应labelname的顺序。
'''

path = '..\\数据集2\\post2012'
labelname = os.listdir(path)   # 获取该路径下的子文件名
audio_data_set = dict()


sample_num = []
for i in range(len(labelname)):
    subpath = path + '\\' + labelname[i]
    subfilename = os.listdir(subpath)
    sample_num.append(len(subfilename))   # 统计文件数量，也可以认为是统计样本数量
    list = []
    for j in range(len(subfilename)):
        audio_data, sample_rate = librosa.load(
            subpath + '\\' + subfilename[j], sr=22000, mono=True, res_type='kaiser_best')
        list.append(audio_data.tolist())
        print(i, j, sample_rate)
        # plt.plot(audio_data)
        # plt.show()
    audio_data_set[labelname[i]] = list


with open('instrument_data.json', 'w', encoding='utf-8') as f_obj:
    json.dump(audio_data_set, f_obj)
