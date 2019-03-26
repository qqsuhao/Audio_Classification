# -*- coding:utf8 -*-
# @TIME     : 2019/3/25 20:54
# @Author   : SuHao
# @File     : load_instrument_data.py

import numpy as np
import librosa
# import pyAudioAnalysis as paa
# from pyAudioAnalysis import audioBasicIO as io
import os

'''
文件夹“数据集2”下有多个子文件，其文件名为每种乐器的名字，因此也是标签。
每个乐器文件夹里有多个音频文件因此每个乐器文件夹需要用一个二维数组来存放数据。
依次读取每个乐器文件夹，然后每个文件夹产生的二维数组存入一个列表，顺序对应labelname的顺序。
'''

path = '..\\数据集2\\pre2012'
labelname = os.listdir(path)   # 获取该路径下的子文件名称
audio_data_set = {}
for i in range(len(labelname)):
    subpath = path + '\\' + labelname[i]
    subfilename = os.listdir(subpath)
    list = []
    for j in range(len(subfilename)):
        audio_data, sample_rate = librosa.load(subpath + '\\' + subfilename[j], sr=None, mono=True, res_type='scipy')
        list.append(audio_data)
        print(i, j, sample_rate)
    audio_data_set[labelname[i]] = list

