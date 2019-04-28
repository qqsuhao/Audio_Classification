# -*- coding:utf8 -*-
# @TIME     : 2019/4/25 16:01
# @Author   : SuHao
# @File     : timbral_feature.py

import scipy.signal as signal
import scipy.fftpack as fftpack
import numpy as np
import librosa as lib
import pyAudioAnalysis.audioFeatureExtraction as pyaudio
import frft
from librosa import filters
import visualization as visual
import warnings
import preprocessing
import copy
from statsmodels.tsa import stattools
import feature_extraction as fe

# 时间序列的自相关函数 已检查
def acf_fundamental_freq(x, fs, fmin, fmax):
    y = copy.deepcopy(x)        # 这里之前有错误
    y = preprocessing.avoid_overlap(y, N=100, f=fmax+100, fs=fs, plot=False)  # fmax+100为的是留出一些裕度，因为低通滤波器不理想
    # time_series = preprocessing.downsample(time_series, fs, 4410)
    nmin = int(fs / fmax)
    nmax = int(fs / fmin)
    acf = stattools.acf(y,  nlags=nmax)
    f0 = fs / (np.argmax(acf[nmin:]) + nmin)
    return f0


# 帧矩阵的自相关函数 已检查
def acf_fundamental_freq_frames(frames, fs, fmin, fmax):
    '''
    :param frames: 帧
    :param fs: 采样率
    :param fmin: 最小频率范围
    :param fmax: 最大频率范围
    :return: (frames.shape[1], )
    '''
    f0 = np.zeros((frames.shape[1], ))
    for i in range(frames.shape[1]):
        f0[i] = acf_fundamental_freq(frames[:, i], fs, fmin, fmax)
    return f0


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


# 谐波提取 没有详细检查
def harmonics(frames, fs, S, f, nfft, fmin=50, fmax=500,  nht=0.15):
    '''
    :param frames: 帧
    :param fs:
    :param S: stft
    :param f: stft的频率
    :param nfft: fft点数
    :param fmin:
    :param fmax:
    :param nht: 0.15
    :return: 两个列表，列表元素为存放谐波频率的array，长度可能都不一样
    '''
    S, f = mystft(frames, fs, 8192)
    y = np.abs(S)
    f0 = acf_fundamental_freq_frames(frames=frames, fmin=fmin, fmax=fmax, fs=fs)
    co = f0*nfft/fs
    harm_freq = []
    harm_mag = []
    for i in range(frames.shape[1]):
        if f0[i] == 0:
            continue
        tmp_freq = np.zeros((int(0.5*fs/f0[i]), ))
        tmp_mag = np.zeros((int(0.5*fs/f0[i]), ))
        for h in range(1, int(0.5*fs/f0[i])+1):
            a0 = int(np.floor((h - nht) * co[i]))
            b0 = int(np.ceil((h + nht) * co[i]))
            if b0 > len(f):
                b0 = len(f)  # 防止b0超出数组范围
            point = a0 + np.argmax(y[a0:b0, i])
            tmp_freq[h-1] = f[point]
            tmp_mag[h-1] = y[point, i]
        harm_freq.append(tmp_freq)
        harm_mag.append(tmp_mag)
    return harm_freq, harm_mag


# 没有详细检查
def harmonic_spectral_centroid(harm_freq, harm_mag):
    warnings.filterwarnings("ignore")
    hsc = np.zeros((len(harm_mag), ))
    for i in range(len(harm_mag)):
        freq = harm_freq[i]
        mag = harm_mag[i]
        hsc[i] = (freq*mag).sum() / mag.sum()   # 即使freq和mag部分值可能为0，也不影响计算
    warnings.filterwarnings("default")
    return hsc.reshape(1, len(harm_mag))


# 没有详细检查
def harmonic_spectral_deviation(harm_mag):
    '''
    :param harm_mag:
    :param flag: 提取这个特征要至少有3个谐波，为了防止出错而设
    :return: 长度等于帧的个数的一维数组
    '''
    warnings.filterwarnings("ignore")
    hsd = np.zeros((len(harm_mag), ))
    for i in range(len(harm_mag)):
        mag = harm_mag[i]
        se = np.zeros((len(mag), ))
        rows = len(mag)
        se[0] = 0.5*(mag[0] + mag[1])
        se[1: rows-1] = (mag[0:rows-2] + mag[1:rows-1] + mag[2:rows]) / 3
        se[rows-1] = 0.5*(mag[-2] + mag[-1])
        hsd[i] = np.abs((np.log10(mag) - np.log10(se))).sum() / np.log10(mag).sum()
    warnings.filterwarnings("default")
    return hsd.reshape(1, len(harm_mag))


# 没有详细检查
def harmonic_spectral_spread(hsc, harm_freq, harm_mag):
    warnings.filterwarnings("ignore")
    hss = np.zeros((len(harm_mag), ))
    for i in range(len(harm_mag)):
        mag = harm_mag[i]
        freq = harm_freq[i]
        hss[i] = ((freq-hsc[0, i])**2 * mag**2).sum() / (mag**2).sum()
    hss = hss.reshape(1, len(harm_mag))
    hss = np.sqrt(hss) / hsc
    warnings.filterwarnings("default")
    return hss.reshape(1, len(harm_mag))


# 没有详细检查
def harmonic_spectral_variation(harm_mag):
    warnings.filterwarnings("ignore")       # 由于计算过程中可能会出现nan，为了防止出现警告
    cols = len(harm_mag)
    hsv = np.zeros((len(harm_mag), ))
    for i in range(cols-1):
        mag = harm_mag[i]
        mag2 = harm_mag[i+1]
        long = min(len(mag), len(mag2))
        mag = mag[0: long]
        mag2 = mag2[0: long]
        a = np.sqrt((mag**2).sum())
        b = np.sqrt((mag2**2).sum())
        hsv[i] = 1 - (mag * mag2).sum() / (a*b)
    warnings.filterwarnings("default")
    return hsv.reshape(1, len(harm_mag))
