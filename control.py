# -*- coding:utf8 -*-
# @TIME     : 2019/3/31 22:58
# @Author   : SuHao
# @File     : control.py

from load_and_preprocess import *
from reload_and_feature import *
from reload_and_classify import *


saveprojectpath = '..\\仿真结果\\cello_and_viola'
path = '..\\cello_and_viola'  # 数据集路径
# saveprojectpath = '..\\仿真结果\\17_18'
# path = '..\\17_18'  # 数据集路径
downsample_rate = 44100
frame_length = int(0.03 * downsample_rate)  # 30ms  窗口长度不要太小，否则会有警告：mfcc映射以后的一些区间是空的。
frame_overlap = frame_length // 2
test_rate = 0.4
feature_type = [12]
'''
0.stft
1.zero_crossing_rate
2.energy
3.entropy_of_energy
4.spectral_centroid_spread
5.spectral_entropy
6.spectral_flux
7.spectral_rolloff
8.bandwidth
9.mfccs
10.rms
11.stfrft
12.frft_mfcc
13.Harmonics

'''
savedata = saveprojectpath + '\\data'
savepic = saveprojectpath + '\\pic'
savetestdata = savedata + '\\' + 'test_data'
savepreprocess = savedata + '\\' + 'preprocessing_result.csv'
savefeature = savedata + '\\' + 'savefeature'


params = {'saveprojectpath': saveprojectpath,
          'savedata': savedata,
          'savepic': savepic,
          'savetestdata': savetestdata,
          'savepreprocess': savepreprocess,
          'savefeature': savefeature,
          'path': path,
          'downsample_rate': downsample_rate,
          'frame_length': frame_length,
          'frame_overlap': frame_overlap,
          'test_rate': test_rate}

# _ = load_and_preprocess(amphasis=False, **params)
# reload_and_feature(picall=False, feature_type=feature_type, **params)
accuracy = []
for i in range(20):
    accuracy.append(reload_and_classify(order=i, PCAornot=False, **params))
    print(accuracy[i])
print(sum(accuracy) / 20)
