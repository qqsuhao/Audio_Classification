# -*- coding:utf8 -*-
# @TIME     : 2019/3/31 22:58
# @Author   : SuHao
# @File     : control.py

from load_and_preprocess import *
from reload_and_feature import *
from reload_and_classify import *


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
test_rate = 0.98

params = {'saveprojectpath': saveprojectpath,
          'savedata': savedata,
          'savepic': savepic,
          'savetestdata': savetestdata,
          'savepreprocess': savepreprocess,
          'savetrainfeature': savetrainfeature,
          'savetestfeature': savetestfeature,
          'path': path,
          'downsample_rate': downsample_rate,
          'frame_length': frame_length,
          'frame_overlap': frame_overlap,
          'test_rate': test_rate}

_ = load_and_preprocess(**params)
reload_and_feature(**params)
reload_and_classify(**params)
