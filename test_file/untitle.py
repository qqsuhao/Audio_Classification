# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:20:39 2019

@author: HAO SU
"""

import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import librosa as lib
from preprocessing import frame
from preprocess_filter import *


def stft(x, pic=None, **params):
    f, t, zxx = signal.stft(x, **params)
    if pic is not None:
        visual.stft_specgram(f, t, zxx, picname=pic + '_stft')
    return f, t, zxx


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
    stft = stft / (frames.shape[0] / 2)  # 归一化
    f = np.linspace(0, np.pi, stft.shape[0], endpoint=True) * fs / np.pi / 2
    return stft, f

def stft_specgram(f, t, zxx, picname=None):
    plt.figure()
    plt.pcolormesh(t, f, (np.abs(zxx)))
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()


t=np.arange(0, 0.1, 0.001)
x = np.sin(2*np.pi*200 * t) + np.sin(2*np.pi*50*t)
# frames = frame(x, 1000, 100)
# f1, _, S1 = stft(x, pic=None, fs=10000, nperseg=1000,noverlap=100, nfft=8192)
# S2, f2 = mystft(frames, 10000, 8192)
# stft_specgram(f2, np.arange(0,S2.shape[1]), S2, picname=None)

b, a = signal.butter(N=20, Wn=100, btype='low', output='ba', fs=1000)
# out = signal.lfilter(b, a, x)
out = signal.filtfilt(b, a, x)
plt.plot(out)
plt.show()