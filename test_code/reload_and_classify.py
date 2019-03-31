# -*- coding:utf8 -*-
# @TIME     : 2019/3/30 22:06
# @Author   : SuHao
# @File     : reload_and_classify.py

import csv
import numpy as np
import pandas as pd


# saveprojectpath = '..\\仿真结果\\boy_and_girl_without_pre_amphasis_hilbert'
# savedata = saveprojectpath + '\\data'
# savepic = saveprojectpath + '\\pic'
# savepreprocess = savedata + '\\' + 'preprocessing_result.csv'
# savefeature = savedata + '\\' + 'feature_result.csv'
# path = '..\\boy_and_girl'  # 数据集路径
# downsample_rate = 8000
# frame_length = int(0.02 * downsample_rate)  # 20ms
# frame_overlap = frame_length // 2


# features = pd.read_csv(savefeature, encoding='utf-8')


# datafile = open(savepreprocess, encoding='utf-8')
# features = csv.reader(datafile)

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

