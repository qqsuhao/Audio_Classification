# -*- coding:utf8 -*-
# @TIME     : 2019/3/29 9:31
# @Author   : SuHao
# @File     : feature_extraction.py

import scipy.signal as signal
import scipy.fftpack as fftpack
import numpy as np
import librosa as lib
import pyAudioAnalysis.audioFeatureExtraction as pyaudio
import matplotlib.pyplot as plt
import frft
from librosa import filters

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


'''
def stft(x, pic=None, **params):
    '''
    boxcar, triang, blackman, hamming, hann,bartlett, flattop, parzen, bohman, blackmanharris,
    nuttall, barthann, kaiser (needs beta),gaussian (needs standard deviation),
    general_gaussian (needs power, width), slepian (needs width),
    dpss (needs normalized half-bandwidth), chebwin (needs attenuation),
    exponential (needs decay scale), tukey (needs taper fraction)
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
    plt.figure()
    plt.pcolormesh(t, f, (np.abs(zxx)))
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    if pic is not None:
        plt.savefig(str(pic) + '.jpg')
    plt.clf()
    plt.close()
    return f, t, zxx


def fft_single_spectral(x):
    '''
    :param: x:输入信号
    :return: 偶数点fft的单边谱。
    '''
    if len(2) % 2 is not 0:
        raise Exception('fft wrong')
        return
    X = fftpack.fft(x)
    return X[1:len(x) / 2 + 1]


def zero_crossing_rate(frames):
    # 特征列向量
    # array也是可迭代的，按照行迭代
    # 使用map（func, array)相当于for循环遍历行
    # frame每一列是一帧
    feature = np.array(list(map(pyaudio.stZCR, frames.T)))
    if feature.shape[0] == frames.shape[1]:
        return np.array([feature])
    else:
        raise Exception("zero_crossing_rate wrong")


def energy(frames):
    feature = np.array(list(map(pyaudio.stEnergy, frames.T)))
    if feature.shape[0] == frames.shape[1]:
        return np.array([feature])
    else:
        raise Exception("energy wrong")


def entropy_of_energy(frames, n_short_blocks=10):
    feature = np.array(list(map(pyaudio.stEnergyEntropy, frames.T,
                                n_short_blocks * np.ones((frames.shape[1], )).astype('int'))))
    if feature.shape[0] == frames.shape[1]:
        return np.array([feature])
    else:
        raise Exception("entropy_of_energy wrong")


def spectral_centroid_spread(X, fs):
    '''
    :param X:库函数的输入参数是fft单边谱,使用fft函数求得频谱以后
            如果点数为N，取fft结果的1：N/2+1
            X的每一列是一个帧对应的单边谱
            X可以直接用STFT的结果
    :return:
    '''
    y = np.abs(X)
    feature = np.array(
        list(map(pyaudio.stSpectralCentroidAndSpread, y.T, fs * np.ones((y.shape[1],)))))
    if feature.shape[0] == y.shape[1]:
        return np.array([feature[:, 0]]), np.array([feature[:, 1]])
    else:
        raise Exception("spectral_centroid_spread wrong")


def spectral_entropy(X, n_short_blocks=10):
    y = np.abs(X)
    feature = np.array(list(map(pyaudio.stSpectralEntropy,
                                y.T, n_short_blocks * np.ones((y.shape[1],)).astype('int'))))
    if feature.shape[0] == y.shape[1]:
        return np.array([feature])
    else:
        raise Exception("spectral_entropy wrong")


def spectral_flux(X):
    y1 = np.abs(X)
    y2 = np.concatenate([np.zeros((X.shape[0], 1)), y1[:, 1:]], axis=1)
    feature = np.array(
        list(map(pyaudio.stSpectralFlux, y1.T, y2.T)))
    if feature.shape[0] == y1.shape[1]:
        return np.array([feature])
    else:
        raise Exception("spectral_flux wrong")
    pass


def spectral_rolloff(X, c, fs):
    c = np.array(c).astype('float32')
    y = np.abs(X)
    feature = np.array(list(map(pyaudio.stSpectralRollOff, y.T,
                                c * np.ones((y.shape[1],)), fs * np.ones((y.shape[1],)))))
    if feature.shape[0] == y.shape[1]:
        return np.array([feature])
    else:
        raise Exception("spectral_flux wrong")
    pass


def bandwidth(X, freq, norm=True, p=2):
    '''
    :param X:
    :param params: sr:采样频率；n_fft：fft长度，4；hop_length：窗口移动步长；
                   X=频谱图，freq=频谱图对应的频率, centroid=None, norm=True, p=2
    :return:
    '''
    y = np.abs(X)
    feature = lib.feature.spectral_bandwidth(S=y, freq=freq, norm=norm, p=p)
    return feature


def mfccs(X, fs, nfft, n_mels=128, n_mfcc=13, dct_type=2):
    '''
    y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho'
    '''
    # 每一个帧有13个MFCC特征
    y = np.abs(X)**2
    S = lib.feature.melspectrogram(S=y, sr=fs, n_fft=nfft, n_mels=128)
    feature = lib.feature.mfcc(S=lib.power_to_db(S), sr=fs, n_mfcc=n_mfcc, dct_type=dct_type)
    return feature


def rms(X):
    '''
    :param X: stft频谱(复数)
    :return:
    '''
    y = np.abs(X)
    return lib.feature.rms(S=y)


def chroma_vector():
    pass


def chroma_deviation():
    pass


def stfrft(frames, p=1):
    stfrft = np.array(list(map(frft.frft, frames.T, p*np.ones((frames.shape[1], )).astype('float32'))))
    if frames.shape[0] % 2 == 0:
        stfrft = stfrft[:, frames.shape[0]/2-1:]
    else:
        stfrft = stfrft[:, frames.shape[0]//2+1:]
    stfrft = stfrft.T
    if stfrft.shape[1] == frames.shape[1]:
        return stfrft
    else:
        raise Exception("stfrft wrong")


def frft_MFCC(S, fs, n_fft, n_mfcc=13, n_mels=128, dct_type=2, norm='ortho', power=2):

    # Build a Mel filter
    S = np.abs(S)**power
    mel_basis = filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=None, htk=False, norm=1)
    melspectrogram = np.dot(mel_basis, S)
    S_db = lib.core.power_to_db(melspectrogram)
    return fftpack.dct(S_db, axis=0, type=dct_type, norm=norm)[:n_mfcc]
