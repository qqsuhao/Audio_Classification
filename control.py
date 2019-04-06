# -*- coding:utf8 -*-
# @TIME     : 2019/3/31 22:58
# @Author   : SuHao
# @File     : control.py

from load_and_preprocess import *
from reload_and_feature import *
from reload_and_classify import *


saveprojectpath = '..\\仿真结果\\music_without_pre_amphasis_hilbert'
path = '..\\数据集2\\post2012'  # 数据集路径
downsample_rate = 22050
frame_length = int(0.03 * downsample_rate)  # 30ms  窗口长度不要太小，否则会有警告：mfcc映射以后的一些区间是空的。
frame_overlap = frame_length // 2
test_rate = 0.3
feature_type = [1,2,3,4,5,6,7,8,9,10]
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
# reload_and_feature(feature_type, **params)
accuracy = []
for i in range(100):
    accuracy.append(reload_and_classify(order=i, **params))
    print(accuracy[i])
print(sum(accuracy) / 100)
