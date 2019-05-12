# -*- coding:utf8 -*-
# @TIME     : 2019/5/7 9:35
# @Author   : SuHao
# @File     : control_vliad.py

from load_and_preprocess import *
from reload_and_feature import *
# from reload_and_classify import *
from load_and_classify1 import *
# from load_and_feature_2 import *


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
13.harmonic ratio and fundalmental_freq
14.chroma_stft
15.log attack time
16.temporal centroid
17.harmonic spectral CDSV
18.pitches mag CDSV(指定频率范围的音色特征)
19.1-order delta of mfccs
20.2-order delta of mfccs

'''


def control(**params):
    saveprojectpath = params['saveprojectpath']     # 仿真结果数据存放路径
    path = params['path']                           # 数据集路径
    downsample_rate = params['downsample_rate']     # 降采样率
    frame_time = params['frame_time']               # 帧长 ms
    overlap_time = params['overlap_time']           # 帧重叠长度 ms
    test_rate = params['test_rate']                 # 测试样本占比
    feature_type = params['feature_type']           # 选择要计算的特征的类型，是一个list
    amphasis = params['amphasis']                   # 预加重，bool
    down = params['down']                           # 降采样， bool
    clip = params['clip']                           # silence remove， bool
    factor = params['factor']                       # 希尔伯特滤波器门限
    nmel = params['neml']                           # 梅尔频率个数
    order_frft = params['order_frft']               # 分数阶傅里叶变换阶数，是一个数组，不同的样本类型阶数可能不同
    nmfcc = params['nmfcc']                         # mfcc系数的个数
    picall = params['picall']                       # 绘制特征的图像， bool
    average = params['average']                     # 使用统计平均作为特征， bool
    feature_select = params['feature_select']       # 特征选择， False或者list
    PCAornot = params['PCAornot']                   # 是否绘制PCA图像， bool
    neighbors = params['neighbors']                 # KNN最近邻的数量
    metric = params['metric']                       # 距离度量的类型：'euclidean' 'manhattan' 'minkowski' 'mahalanobis' 'chebyshev'
    weight_type = params['weight_type']             # 距离加权的类型：'distance' 'uniform' 'gaussian'
    gaussian_bias = params['gaussian_bias']         # 径向基函数的标准差
    dist_p = params['dist_p']                       # 使用闵可夫斯基距离的附属参数
    pic_i = params['pic_i']                         # 用于调参时将结果存入不同的文件
    savefeature_i = params['savefeature_i']         # 用于存放不同的特征提取结果
    feature_reduce = params['feature_reduce']       # 特征选择的特征个数
    valid_k = params['valid_k']                     # 交叉验证折数，0表示使用留出法


    frame_length = int(frame_time / 1000 * downsample_rate)
    frame_overlap = int(overlap_time / 1000 * downsample_rate)

    savedata = saveprojectpath + '\\data'
    savepic = saveprojectpath + '\\pic' + str(pic_i)
    savetestdata = savedata + '\\' + 'test_data'
    savepreprocess = savedata + '\\' + 'preprocessing_result.csv'
    savefeature = savedata + '\\' + 'savefeature' + str(savefeature_i)
    params_path = {'saveprojectpath': saveprojectpath,          # 仿真结果保存文件夹
                   'savedata': savedata,                             # 缓存数据保存
                   'savepic': savepic,                               # 仿真图片保存
                   'savetestdata': savetestdata,                     # savedata的子文件夹，存放测试数据
                   'savepreprocess': savepreprocess,                 # savedata的子文件，cvs，存放预处理数据
                   'savefeature': savefeature,                       # savedata的子文件夹，存放特征提取的结果
                   'path': path,                                     # 数据集的路径
                   'downsample_rate': downsample_rate,               # 降采样率
                   'frame_time': frame_time,
                   'frame_length': frame_length,
                   'frame_overlap': frame_overlap,
                   'test_rate': test_rate}

    # _ = load_and_preprocess(
    #     amphasis=amphasis,                # bool，是否进行预加重
    #     down=down,                        # bool，是否进行降采样
    #     clip=clip,                        # bool， 是否进行silence remove
    #     factor=factor,                    # 0-1, silence remove门限
    #     **params_path)
    # reload_and_feature(
    # picall=picall,                    # 是否进行绘制所有特征的图像
    # feature_type=feature_type,        # 特征类型
    # average=average,                  # 是否使用所有帧的统计量作为特征而不使用每个帧
    # nmel=nmel,                        # 梅尔频率的个数
    # order_frft=order_frft,            # frft的阶数
    # nmfcc=nmfcc,                      # mfcc的系数个数
    # **params_path)
    accuracy = []                           # 存放每次测试的准确率
    X, Y = reload(feature_reduce=feature_reduce,
                   feature_type=feature_type,
                   feature_select=feature_select,
                   order=0,
                   PCAornot=PCAornot,
                   average=average,
                   neighbors=neighbors,
                   metric=metric,
                   weight_type=weight_type,
                   gaussian_bias=gaussian_bias,
                   dist_p=dist_p,
                   **params_path)
    # N = 50
    # for i in range(N):
    #     accuracy.append(
    #          classify( X=X, Y=Y, valid_k=valid_k,
    #             feature_type=feature_type,
    #             feature_select=feature_select,
    #             order=i,
    #             PCAornot=PCAornot,
    #             average=average,
    #             neighbors=neighbors,
    #             metric=metric,
    #             weight_type=weight_type,
    #             gaussian_bias=gaussian_bias,
    #             dist_p=dist_p,
    #             **params_path))
    #     print(accuracy[i])
    #
    # with open(savepic+'\\params.txt', 'w') as f:
    #     for key, value in params.items():
    #         f.write(key)
    #         f.write(': ')
    #         f.write(str(value))
    #         f.write('\n')
    #     f.write('accuracy: ')
    #     f.write(str(sum(accuracy) / N))
    #
    # return sum(accuracy) / N


best = {'saveprojectpath': '..\\仿真结果\\post2012_best',
        'path': '..\\数据集2\\post2012',
        'downsample_rate': 44100,
        'frame_time': 30,
        'overlap_time': 1,
        'test_rate': 0.3,
        'feature_type': [1,2,3,4,5,6,7,8,9,10,12,14,15,16,17,18,19,20],
        'amphasis': False,
        'down': False,
        'clip': False,
        'factor': 0.1,
        'neml': 128,
        'order_frft': [0.94 for i in range(19)],
        'nmfcc': 13,
        'picall': False,
        'average': True,
        'feature_select': [1,2,3,4,5,6,7,8,9,10,12,15,16,17,19],
        'PCAornot': False,
        'neighbors': 1,
        'metric': 'manhattan',
        'weight_type': 'distance',          # 距离加权的类型：'distance' 'uniform' 'gaussian'
        'gaussian_bias': 2,                 # 径向基函数的标准差
        'dist_p': 1,
        'pic_i': 1000,
        'savefeature_i': 1,
        'feature_reduce': 85,
        'valid_k': 3,
        'index': 'feature selection'}


#best = {'saveprojectpath': '..\\仿真结果\\post2012_preprocess',
#        'path': '..\\数据集2\\post2012',
#        'downsample_rate': 22050,
#        'frame_time': 30,
#        'overlap_time': 1,
#        'test_rate': 0.3,
#        'feature_type': [1,2,3,4,5,6,7,8,9,10,12,14,15,16,17,18,19,20],
#        'amphasis': False,
#        'down': True,
#        'clip': False,
#        'factor': 0.1,
#        'neml': 128,
#        'order_frft': [0.94 for i in range(19)],
#        'nmfcc': 13,
#        'picall': False,
#        'average': True,
#        'feature_select': [1,2,3,4,5,6,7,8,9,10,15,16,19],
#        'PCAornot': False,
#        'neighbors': 1,
#        'metric': 'manhattan',
#        'weight_type': 'distance',          # 距离加权的类型：'distance' 'uniform' 'gaussian'
#        'gaussian_bias': 2,                 # 径向基函数的标准差
#        'dist_p': 1,
#        'pic_i': 1,
#        'savefeature_i': 1,
#        'index': 'none'}

# type = [[1,2,3,4,5,6,7,8,9,10],      # 基础 + mfcc
#         [1,2,3,4,5,6,7,8,9,10,18],   # 基础 + mfcc + 之前错误的谐波
#         [1,2,3,4,5,6,7,8,9,10,19],   # 基础 + mfcc + 一阶mfcc
#         [1,2,3,4,5,6,7,8,9,10,19,20], # 基础 + mfcc + 一阶mfcc + 二阶
#         [1,2,3,4,5,6,7,8,9,10,15,16,17] # 基础 + mfcc +
#         [1,2,3,4,5,6,7,8,9,10,15,16,17]  # 基础 + mfcc + 音色特征
#         ]
# pici = [11,12,13,14,15]
# for i in range(5):
#     best['feature_type'] = type[i]
#     best['pic_i'] = pici[i]
control(**best)

'''

'''