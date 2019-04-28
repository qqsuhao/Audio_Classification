# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:20:39 2019

@author: HAO SU
"""

import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy import fftpack
import librosa as lib
from preprocessing import frame
from librosa import util
import numpy as np
import scipy.fftpack as fft
from scipy.signal import get_window

def stft(x, pic=None, **params):
    '''
    :param x: 输入信号
    :param params: {fs:采样频率；
                    window:窗。默认为汉明窗；
                    nperseg： 每个段的长度，默认为256，
                    noverlap:重叠的点数。指定值时需要满足COLA约束。默认是窗长的一半，
                    nfft：fft长度，
                    detrend：（str、function或False）指定如何去趋势，默认为Flase，不去趋势。
                    return_onesided：默认为True，返回单边谱。
                    boundary：
                    padded：是否对时间序列进行填充0（当长度不够的时候），
                    axis：}
    :return: f:采样频率数组；t:段时间数组；Zxx:STFT结果
    '''
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


def libstft(y, fs, n_fft=2048, hop_length=None, win_length=None, window='hann',
            center=None, dtype=np.complex64, pad_mode='reflect'):
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = util.pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    util.valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    # Window the time series.
    y_frames = util.frame(y, frame_length=win_length, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(util.MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]]
    f = np.linspace(0, np.pi, stft_matrix.shape[0], endpoint=True) * fs / np.pi / 2
    return stft_matrix, f


def stft_specgram(f, t, zxx, picname=None):
    plt.figure()
    plt.pcolormesh(t, f, (np.abs(zxx)))
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()


t=np.arange(0, 1, 0.0001)
x = np.sin(2*np.pi*200 * t) + np.sin(2*np.pi*50*t)
frames = frame(x, 1000, 100)
f1, _, S1 = stft(x, pic=None, fs=10000, nperseg=1000, noverlap=1000-100, nfft=8192, boundary=None, padded=False)
S2, f2 = mystft(frames, 10000, 8192)
# S3, f3 = libstft(x, fs=10000, n_fft=8192, hop_length=100, win_length=1000,center=False, dtype=np.complex64, pad_mode='reflect')
stft_specgram(f2, np.arange(0,S2.shape[1]), S2, picname=None)