# -*- coding:utf8 -*-
# @TIME     : 2019/3/30 20:50
# @Author   : SuHao
# @File     : load_and_preprocess.py

import numpy as np
import librosa
import os
import preprocessing
import csv
import matplotlib.pyplot as plt
import amfm_decompy.pYAAPT as pYAAPT


'''
对于不同的数据集，进行测试时，只需要更改saveprojectpath 和 path
(不包括预处理等其他内容的详细参数设置)
'''


def load_and_preprocess(amphasis,
                        down,
                        clip,
                        factor,
                        saveprojectpath,
                        savedata,
                        savepic,
                        savetestdata,
                        savepreprocess,
                        savefeature,
                        path,
                        downsample_rate,
                        frame_time,
                        frame_length,
                        frame_overlap,
                        test_rate):
    if not os.path.exists('..\\仿真结果'):
        os.mkdir('..\\仿真结果')
    if not os.path.exists(saveprojectpath):
        os.mkdir(saveprojectpath)
    if not os.path.exists(savedata):        # 保存数据
        os.mkdir(savedata)
    if not os.path.exists(savepic):         # 创建保存图片的文件夹
        os.mkdir(savepic)
    if not os.path.exists(savetestdata):    # 创建保存测试数据的文件
        os.mkdir(savetestdata)

    # 写入数据
    with open(savepreprocess, 'w', encoding='utf-8') as csvfile:    # 先创建csv文件，什么都不写入，之后再追加
        writer = csv.writer(csvfile)
    # 读取音频文件目录
    labelname = os.listdir(path)                                    # 获取该路径下的子文件名，也是标签的名字
    for j in range(len(labelname)):
        subpath = path + '\\' + labelname[j]                        # 数据集的子文件路径
        subfilename = os.listdir(subpath)                           # 查看音频文件目录
        for i in range(len(subfilename)):
            audio_data, sample_rate = librosa.load(
                subpath + '\\' + subfilename[i], sr=None, mono=True, res_type='kaiser_best')  # 读取音频文件
            audio_data = librosa.util.normalize(audio_data, norm=np.inf, axis=0, threshold=None, fill=None)
            # audio_data = audio_data[0:8820]
##########################################################################################################################
            if amphasis:
                pre_amphasis = preprocessing.pre_emphasis(audio_data, 0.97,
                                                          pic=None)
                                                          # pic=savepic + '\\' + 'pre_amphasis_'+str(j)+'_'+str(i))
            else:
                pre_amphasis = audio_data

            if down:
                avoid_overlap = preprocessing.avoid_overlap(pre_amphasis,
                                                            N=20,
                                                            f=downsample_rate / 2,
                                                            fs=sample_rate,
                                                            plot=False)
                downsample = preprocessing.downsample(
                    avoid_overlap, sample_rate, downsample_rate)
            else:
                downsample = pre_amphasis

            if clip == 'hilbert':
                silence_remove = preprocessing.silence_remove(
                    x=downsample,
                    limit=np.max(downsample) * factor,
                    fs=downsample_rate,
                    option='hilbert',
                    # pic=savepic + '\\' + 'silence_remove_hilbert_' + str(j)+'_'+str(i))
                    pic=None)
            elif clip == 'HF':
                silence_remove = preprocessing.silence_remove(
                    x=downsample,
                    limit=np.max(downsample) * factor,
                    fs=downsample_rate,
                    option='HF',
                    # pic=savepic + '\\' + 'silence_remove_hilbert_filter_' + str(j)+'_'+str(i))
                    pic=None)
            elif clip == 'filter':
                silence_remove = preprocessing.silence_remove(
                        x=downsample,
                        limit=0.02,
                        option='filter',
                        pic=savepic + '\\' + 'silence_remove_filter_' + str(j)+'_'+str(i),
                        N=10,
                        f=600,
                        fs=downsample_rate)
            else:
                silence_remove = downsample


########################################################################################################################
            # j表示标签， i表示同一标签下的音频文件序号
            buffer = [j] + [i] + list(silence_remove)
            with open(savepreprocess, 'a+', newline='', encoding='utf-8') as csvfile:
                csv_write = csv.writer(csvfile)
                csv_write.writerow(buffer)              # 逐行写入数据

            print('preprocessing:', j, i)

    return labelname
