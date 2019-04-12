# -*- coding: utf-8 -*-
import visualization as visual
import scipy.fftpack as fft
import preprocessing
import os
import numpy as np
import librosa
"""
Created on Wed Apr 10 15:53:52 2019

@author: HAO SU
"""

'''
主要用于测试预处理相关环节，用于毕业报告。主要内容包括：
1.验证抽取频谱没有发生混叠：绘制原信号频谱；无抗混叠下采样频谱；有抗混叠下采样频谱
2.展示预加重前后的波形
3.验证silence remove三种方法的结果
'''


datapath = '..\\数据集2\\pre2012\\ebclarinet\\EbClar.mf.G3B3.aiff'
savefile = '..\\预处理结果用于写报告'
if not os.path.exists(savefile):
    os.mkdir(savefile)
audio_data, sample_rate = librosa.load(
    datapath, sr=None, mono=True, res_type='kaiser_best')  # 读取文件

# 验证抽取
original_spec = np.abs(fft.fftshift((fft.fft(audio_data)))[0:])
visual.picplot(x=np.linspace(0, np.pi, len(original_spec), endpoint=True),
               y=original_spec,
               title='抽取前原信号频谱',
               xlabel='数字频率',
               ylabel='幅度',
               pic=savefile + '抽取前原信号频谱.jpg')

