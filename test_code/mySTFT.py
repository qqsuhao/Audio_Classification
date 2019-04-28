# -*- coding:utf8 -*-
# @TIME     : 2019/4/26 0:29
# @Author   : SuHao
# @File     : mySTFT.py

import numpy as np
import librosa as lib
import preprocessing
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import scipy.signal as signal
import feature_extraction as fe

path = '..\\..\\数据集2\\pre2012\\bflute\\BassFlute.ff.C5B5.aiff'
data, fs = lib.load(path, sr=None, mono=True, res_type='kaiser_best')
frame_length = int(0.03*fs)
frame_overlap = int(0.015*fs)
frames = preprocessing.frame(data, frame_length, frame_overlap)





def mystft(frames, fs, nfft):
    '''
    :param frames: 帧数组，一列为一帧
    :param fs: 采样频率
    :param nfft: fft点数
    :return:
    '''
    def fft_single(x, n):
        X = fftpack.fft(x=x, n=n)
        X = X[0: len(X) // 2 + 1]
        return X
    window = signal.get_window('hamming', frames.shape[0])
    window = window.reshape(len(window), 1)
    window = np.tile(window, (1, frames.shape[1]))
    frames_win = frames * window
    stft = np.array(list(map(fft_single, frames_win.T,
                                nfft*np.ones((frames_win.shape[1], )).astype('int32'))))
    stft = stft.T    # 需要转置
    stft = stft / nfft  # 归一化
    f = np.linspace(0, np.pi, stft.shape[0], endpoint=True) * fs / np.pi / 2
    return stft, f


