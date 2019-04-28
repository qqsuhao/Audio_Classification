# -*- coding:utf8 -*-
# @TIME     : 2019/4/25 23:59
# @Author   : SuHao
# @File     : silence_remove_Kmean.py


import numpy as np
import librosa as lib
import preprocessing
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import scipy.signal as signal

path = '..\\数据集2\\pre2012\\bflute\\BassFlute.ff.C5B5.aiff'
data, fs = lib.load(path, sr=None, mono=True, res_type='kaiser_best')
data = lib.util.normalize(data)
# data = data - np.min(data)
frame_length = int(0.01*fs)
frame_overlap = 100
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
    f = np.linspace(0, np.pi, stft.shape[0], endpoint=True) * fs / np.pi / 2
    return stft, f
stft, f = mystft(frames, fs, nfft=8196)


def spectral_crest_factor(stft):
    y = np.abs(stft)
    # a = np.max(y, axis=0)
    # b = np.abs(stft).sum(axis=0)
    # scf = a/b
    scf = np.zeros((stft.shape[1], ))
    for i in range(stft.shape[1]):
        scf[i] = np.max(y[:, i]) / y[:, i].sum()
    return scf.reshape(1, len(scf))


def spectral_flatness(stft):
    Nw = stft.shape[0]
    y = np.abs(stft)
    # a = np.exp(1 / Nw * np.log(y).sum(axis=0))
    # b = 1 / Nw * y.sum(axis=0)
    # sf = a/b
    sf = np.zeros((stft.shape[1], ))
    for i in range(stft.shape[1]):
        a = np.exp(1 / Nw * np.log(y[:, i]).sum())
        b = 1 / Nw * y[:, i].sum()
        sf[i] = a / b
    return sf.reshape(1, len(sf))


def tonal_power_ratio(stft):
    y = np.abs(stft)**2
    Nw = stft.shape[0]
    tpr = np.zeros((stft.shape[1], ))
    for i in range(stft.shape[1]):
        a = y[:, i]
        select = np.zeros((len(a),))
        for j in range(len(a)-2):
            loc = np.argmin(a[j:j+3])
            select[j+loc] = 1
        b = (a * select).sum()
        c = a.sum()
        tpr[i] = b/c
    return tpr.reshape(1, len(tpr))


scf = spectral_crest_factor(stft)
sf = spectral_flatness(stft)
# tpr = tonal_power_ratio(stft)
plt.figure()
plt.subplot(311)
plt.plot(scf[0, :])
plt.subplot(312)
plt.plot(sf[0, :])
plt.subplot(313)
plt.plot(data)
plt.show()