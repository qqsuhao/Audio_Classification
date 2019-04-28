# -*- coding:utf8 -*-
# @TIME     : 2019/4/21 17:45
# @Author   : SuHao
# @File     : control2.py

# -*- coding:utf8 -*-
# @TIME     : 2019/3/31 22:58
# @Author   : SuHao
# @File     : control.py

from load_and_preprocess import *
from reload_and_feature import *
from reload_and_classify import *


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
18.pitches mag CDSV


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

    # saveprojectpath = '..\\仿真结果\\测试预处理数据集'
    # path = '..\\测试预处理数据集'
    # saveprojectpath = '..\\仿真结果\\post2012'
    # path = '..\\数据集2\\post2012'
    # saveprojectpath = '..\\仿真结果\\cello_and_viola'
    # path = '..\\cello_and_viola'  # 数据集路径
    # saveprojectpath = '..\\仿真结果\\17_18'
    # path = '..\\17_18'  # 数据集路径
    # downsample_rate = 44100
    # frame_time = 30.0
    # 30ms  窗口长度不要太小，否则会有警告：mfcc映射以后的一些区间是空的。
    frame_length = int(frame_time / 1000 * downsample_rate)
    frame_overlap = int(overlap_time / 1000 * downsample_rate)

    savedata = saveprojectpath + '\\data'
    savepic = saveprojectpath + '\\pic' + str(pic_i)
    savetestdata = savedata + '\\' + 'test_data'
    savepreprocess = savedata + '\\' + 'preprocessing_result.csv'
    savefeature = savedata + '\\' + 'savefeature'
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

    _ = load_and_preprocess(
        amphasis=amphasis,                # bool，是否进行预加重
        down=down,                        # bool，是否进行降采样
        clip=clip,                        # bool， 是否进行silence remove
        factor=factor,                    # 0-1, silence remove门限
        **params_path)
    reload_and_feature(
        picall=picall,                    # 是否进行绘制所有特征的图像
        feature_type=feature_type,        # 特征类型
        average=average,                  # 是否使用所有帧的统计量作为特征而不使用每个帧
        nmel=nmel,                        # 梅尔频率的个数
        order_frft=order_frft,            # frft的阶数
        nmfcc=nmfcc,                      # mfcc的系数个数
        **params_path)
    accuracy = []                           # 存放每次测试的准确率
    for i in range(50):
        accuracy.append(
            reload_and_classify(
                feature_type=feature_type,
                feature_select=feature_select,
                order=i,
                PCAornot=PCAornot,
                average=average,
                neighbors=neighbors,
                metric=metric,
                weight_type=weight_type,
                gaussian_bias=gaussian_bias,
                dist_p=dist_p,
                **params_path))
        print(accuracy[i])

    with open(savepic+'\\params.txt', 'w') as f:
        for key, value in params.items():
            f.write(key)
            f.write(': ')
            f.write(str(value))
            f.write('\n')
        f.write('accuracy: ')
        f.write(str(sum(accuracy) / 50))

    return sum(accuracy) / 50


params1 = {'saveprojectpath': '..\\仿真结果\\post2012_9',
           'path': '..\\数据集2\\post2012',
           'downsample_rate': 44100,
           'frame_time': 30,
           'overlap_time': 15,
           'test_rate': 0.3,
           'feature_type': [1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 16, 17],
           'amphasis': False,
           'down': False,
           'clip': False,
           'factor': 0.1,
           'neml': 128,
           'order_frft': [0.95 for i in range(19)],
           'nmfcc': 13,
           'picall': False,
           'average': False,
           'feature_select': False,
           'PCAornot': False,
           'neighbors': 1,
           'index': 'compared with post2012, different at: neighbors, average'}
params2 = {'saveprojectpath': '..\\仿真结果\\post2012_3',
           'path': '..\\数据集2\\post2012',
           'downsample_rate': 44100,
           'frame_time': 30,
           'overlap_time': 15,
           'test_rate': 0.3,
           'feature_type': [1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 16, 17],
           'amphasis': False,
           'down': False,
           'clip': False,
           'factor': 0.1,
           'neml': 128,
           'order_frft': [0.95 for i in range(19)],
           'nmfcc': 13,
           'picall': False,
           'average': True,
           'feature_select': False,
           'PCAornot': False,
           'neighbors': 3,
           'index': 'compared with post2012, different at: neighbors'}
params3 = {'saveprojectpath': '..\\仿真结果\\post2012_4',
           'path': '..\\数据集2\\post2012',
           'downsample_rate': 44100,
           'frame_time': 30,
           'overlap_time': 5,
           'test_rate': 0.3,
           'feature_type': [1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 16, 17],
           'amphasis': False,
           'down': False,
           'clip': False,
           'factor': 0.1,
           'neml': 128,
           'order_frft': [0.95 for i in range(19)],
           'nmfcc': 13,
           'picall': False,
           'average': True,
           'feature_select': False,
           'PCAornot': False,
           'neighbors': 1,
           'index': 'compared with post2012, different at: overlap_time'}
params4 = {'saveprojectpath': '..\\仿真结果\\post2012_5',
           'path': '..\\数据集2\\post2012',
           'downsample_rate': 44100,
           'frame_time': 30,
           'overlap_time': 15,
           'test_rate': 0.3,
           'feature_type': [1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 16, 17],
           'amphasis': False,
           'down': False,
           'clip': False,
           'factor': 0.1,
           'neml': 200,
           'order_frft': [0.95 for i in range(19)],
           'nmfcc': 13,
           'picall': False,
           'average': True,
           'feature_select': False,
           'PCAornot': False,
           'neighbors': 1,
           'index': 'compared with post2012, different at: nmel'}
params5 = {'saveprojectpath': '..\\仿真结果\\post2012_6',
           'path': '..\\数据集2\\post2012',
           'downsample_rate': 44100,
           'frame_time': 30,
           'overlap_time': 15,
           'test_rate': 0.3,
           'feature_type': [1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 16, 17],
           'amphasis': False,
           'down': False,
           'clip': 'HF',
           'factor': 0.1,
           'neml': 128,
           'order_frft': [0.95 for i in range(19)],
           'nmfcc': 13,
           'picall': False,
           'average': True,
           'feature_select': False,
           'PCAornot': False,
           'neighbors': 1,
           'index': 'compared with post2012, different at: clip'}
params6 = {'saveprojectpath': '..\\仿真结果\\post2012_7',
           'path': '..\\数据集2\\post2012',
           'downsample_rate': 44100,
           'frame_time': 30,
           'overlap_time': 15,
           'test_rate': 0.3,
           'feature_type': [1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 16, 17],
           'amphasis': True,
           'down': False,
           'clip': False,
           'factor': 0.1,
           'neml': 128,
           'order_frft': [0.95 for i in range(19)],
           'nmfcc': 13,
           'picall': False,
           'average': True,
           'feature_select': False,
           'PCAornot': False,
           'neighbors': 1,
           'index': 'compared with post2012, different at: amphasis'}
params7 = {'saveprojectpath': '..\\仿真结果\\post2012_8',
           'path': '..\\数据集2\\post2012',
           'downsample_rate': 22050,
           'frame_time': 30,
           'overlap_time': 15,
           'test_rate': 0.3,
           'feature_type': [1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 16, 17],
           'amphasis': False,
           'down': True,
           'clip': False,
           'factor': 0.1,
           'neml': 128,
           'order_frft': [0.95 for i in range(19)],
           'nmfcc': 13,
           'picall': False,
           'average': True,
           'feature_select': False,
           'PCAornot': False,
           'neighbors': 1,
           'index': 'compared with post2012, different at: down'}
params8 = {'saveprojectpath': '..\\仿真结果\\post2012_10',
           'path': '..\\数据集2\\post2012',
           'downsample_rate': 44100,
           'frame_time': 30,
           'overlap_time': 15,
           'test_rate': 0.3,
           'feature_type': [1, 2, 3, 5, 6, 7, 8, 10, 12, 15, 16, 17],
           'amphasis': False,
           'down': False,
           'clip': False,
           'factor': 0.1,
           'neml': 128,
           'order_frft': [0.95 for i in range(19)],
           'nmfcc': 13,
           'picall': False,
           'average': True,
           'feature_select': False,
           'PCAornot': False,
           'neighbors': 1,
           'index': 'compared with post2012, different at: feature_type frft'}
params9 = {'saveprojectpath': '..\\仿真结果\\post2012_11',
           'path': '..\\数据集2\\post2012',
           'downsample_rate': 44100,
           'frame_time': 30,
           'overlap_time': 15,
           'test_rate': 0.3,
           'feature_type': [1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 16, 18],
           'amphasis': False,
           'down': False,
           'clip': False,
           'factor': 0.1,
           'neml': 128,
           'order_frft': [0.95 for i in range(19)],
           'nmfcc': 13,
           'picall': False,
           'average': True,
           'feature_select': False,
           'PCAornot': False,
           'neighbors': 1,
           'index': 'compared with post2012, different at: feature_type 18'}
params12 = {'saveprojectpath': '..\\仿真结果\\post2012_12',
            'path': '..\\数据集2\\post2012',
            'downsample_rate': 44100,
            'frame_time': 35,
            'overlap_time': 1,
            'test_rate': 0.3,
            'feature_type': [1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 16, 17],
            'amphasis': False,
            'down': False,
            'clip': False,
            'factor': 0.1,
            'neml': 128,
            'order_frft': [0.95 for i in range(19)],
            'nmfcc': 13,
            'picall': False,
            'average': True,
            'feature_select': False,
            'PCAornot': False,
            'neighbors': 1,
            'index': 'compared with post2012, different at: overlap_time, frame_time'}
params13 = {'saveprojectpath': '..\\仿真结果\\post2012_13',
            'path': '..\\数据集2\\post2012',
            'downsample_rate': 44100,
            'frame_time': 30,
            'overlap_time': 15,
            'test_rate': 0.3,
            'feature_type': [9, 10, 15, 16, 17],
            'amphasis': False,
            'down': False,
            'clip': False,
            'factor': 0.1,
            'neml': 128,
            'order_frft': [0.95 for i in range(19)],
            'nmfcc': 13,
            'picall': False,
            'average': True,
            'feature_select': False,
            'PCAornot': False,
            'neighbors': 1,
            'index': 'compared with post2012, different at: feature_type'}
params14 = {'saveprojectpath': '..\\仿真结果\\post2012_14',
            'path': '..\\数据集2\\post2012',
            'downsample_rate': 44100,
            'frame_time': 30,
            'overlap_time': 15,
            'test_rate': 0.3,
            'feature_type': [1, 2, 3, 5, 6, 7, 8, 9, 12, 10, 15, 16, 17],
            'amphasis': False,
            'down': False,
            'clip': False,
            'factor': 0.1,
            'neml': 128,
            'order_frft': [0.95 for i in range(19)],
            'nmfcc': 13,
            'picall': False,
            'average': True,
            'feature_select': False,
            'PCAornot': False,
            'neighbors': 1,
            'index': 'compared with post2012, different at: feature_type'}
params15 = {'saveprojectpath': '..\\仿真结果\\post2012_15',
            'path': '..\\数据集2\\post2012',
            'downsample_rate': 44100,
            'frame_time': 35,
            'overlap_time': 1,
            'test_rate': 0.9,
            'feature_type': [1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 16, 17],
            'amphasis': False,
            'down': False,
            'clip': False,
            'factor': 0.1,
            'neml': 128,
            'order_frft': [0.95 for i in range(19)],
            'nmfcc': 13,
            'picall': False,
            'average': True,
            'feature_select': False,
            'PCAornot': False,
            'neighbors': 1,
            'index': 'compared with post2012, different at: testrate, frame_time, overlap_time'}
params16 = {'saveprojectpath': '..\\仿真结果\\post2012_16',
            'path': '..\\数据集2\\post2012',
            'downsample_rate': 44100,
            'frame_time': 30,
            'overlap_time': 15,
            'test_rate': 0.3,
            'feature_type': [1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 16, 17],
            'amphasis': False,
            'down': False,
            'clip': 'hilbert',
            'factor': 0.2,
            'neml': 128,
            'order_frft': [0.95 for i in range(19)],
            'nmfcc': 13,
            'picall': False,
            'average': True,
            'feature_select': False,
            'PCAornot': False,
            'neighbors': 1,
            'index': 'compared with post2012, different at: clip'}
params17 = {'saveprojectpath': '..\\仿真结果\\post2012_17',
            'path': '..\\数据集2\\post2012',
            'downsample_rate': 44100,
            'frame_time': 35,
            'overlap_time': 1,
            'test_rate': 0.3,
            'feature_type': [1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 16, 17],
            'amphasis': False,
            'down': False,
            'clip': False,
            'factor': 0.1,
            'neml': 128,
            'order_frft': [0.95 for i in range(19)],
            'nmfcc': 13,
            'picall': False,
            'average': True,
            'feature_select': False,
            'PCAornot': False,
            'neighbors': 3,
            'metric': 'manhattan',
            'index': 'compared with post2012, different at: usecost and other best'}
params20 = {'saveprojectpath': '..\\仿真结果\\post2012_20',
            'path': '..\\数据集2\\post2012',
            'downsample_rate': 44100,
            'frame_time': 35,
            'overlap_time': 1,
            'test_rate': 0.3,
            'feature_type': [1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 16, 17],
            'amphasis': False,
            'down': False,
            'clip': False,
            'factor': 0.1,
            'neml': 128,
            'order_frft': [0.95 for i in range(19)],
            'nmfcc': 13,
            'picall': False,
            'average': True,
            'feature_select': False,
            'PCAornot': False,
            'neighbors': 3,
            'metric': 'manhattan',
            'weight_type': 'gaussian',          # 距离加权的类型：'distance' 'uniform' 'gaussian'
            'gaussian_bias': 2,                 # 径向基函数的标准差
            'dist_p': 1,
            'pic_i': 1,
            'index': 'compared with post2012, different at: neighbors, metric, weight_type, gaussian_bias, dist_p'}

# pic_i_list = 0
# metric_list = [['euclidean', 1], ['manhattan', 1], ['minkowski', 3],
#                ['minkowski', 4], ['mahalanobis', 1], ['chebyshev', 1]]
# weight_type_list = [['uniform', 1], ['distance', 1], ['gaussian', 0.1], ['gaussian', 0.5],
#                     ['gaussian', 1], ['gaussian', 2], ['gaussian', 3], ['gaussian', 4], ['gaussian', 5]]
# neighbors_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]



# for i in range(len(weight_type_list)):
#     table = np.zeros((len(neighbors_list), len(metric_list)))
#     params20['weight_type'] = weight_type_list[i][0]
#     params20['gaussian_bias'] = weight_type_list[i][1]
#     for j in range(len(metric_list)):
#         params20['metric'] = metric_list[j][0]
#         params20['dist_p'] = metric_list[j][1]
#         for k in range(len(neighbors_list)):
#             params20['pic_i'] = i*100+j*10+k
#             saveprojectpath = params20['saveprojectpath']
#             if not os.path.exists(saveprojectpath + '\\pic' + str(i*100+j*10+k)):
#                 os.mkdir(saveprojectpath + '\\pic' + str(i*100+j*10+k))          # 创建保存特征结果的文件
#             params20['neighbors'] = neighbors_list[k]
#             accuracy = control(**params20)
#             table[k, j] = accuracy
#             print(i, j, k)
#     with open(params20['saveprojectpath']+'\\'+weight_type_list[i][0]+str(weight_type_list[i][1])+'.csv',
#               'w', encoding='utf-8', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerow(metric_list)
#         csv_writer.writerows(table)

params21 = {'saveprojectpath': '..\\仿真结果\\post2012_22',
            'path': '..\\数据集2\\post2012',
            'downsample_rate': 44100,
            'frame_time': 35,
            'overlap_time': 1,
            'test_rate': 0.3,
            'feature_type': [1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 16, 17],
            'amphasis': False,
            'down': False,
            'clip': False,
            'factor': 0.1,
            'neml': 128,
            'order_frft': [0.95 for i in range(19)],
            'nmfcc': 13,
            'picall': False,
            'average': True,
            'feature_select': False,
            'PCAornot': False,
            'neighbors': 3,
            'metric': 'manhattan',
            'weight_type': 'gaussian',          # 距离加权的类型：'distance' 'uniform' 'gaussian'
            'gaussian_bias': 2,                 # 径向基函数的标准差
            'dist_p': 1,
            'pic_i': 1,
            'index': 'compared with post2012, different at: neighbors, metric, weight_type, gaussian_bias, dist_p'}

pic_i_list = 0
weight_type_list = [['gaussian', 1], ['gaussian', 1.2], ['gaussian', 1.4], ['gaussian', 1.6], ['gaussian', 1.8],
                    ['gaussian', 2], ['gaussian', 2.2], ['gaussian', 2.4], ['gaussian', 2.6], ['gaussian', 2.8],
                    ['gaussian',3.0]]
neighbors_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]


# table = np.zeros((len(neighbors_list), len(weight_type_list)))
# for i in range(len(weight_type_list)):
#     params21['weight_type'] = weight_type_list[i][0]
#     params21['gaussian_bias'] = weight_type_list[i][1]
#     for k in range(len(neighbors_list)):
#         params21['pic_i'] = i*10+k
#         saveprojectpath = params21['saveprojectpath']
#         if not os.path.exists(saveprojectpath + '\\pic' + str(i*10+k)):
#             os.mkdir(saveprojectpath + '\\pic' + str(i*10+k))          # 创建保存特征结果的文件
#         params21['neighbors'] = neighbors_list[k]
#         accuracy = control(**params21)
#         table[k, i] = accuracy
#         print(i, k)
# with open(params21['saveprojectpath']+'\\result'+'.csv',
#           'w', encoding='utf-8', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     csv_writer.writerow(weight_type_list)
#     csv_writer.writerows(table)



