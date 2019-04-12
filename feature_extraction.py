# -*- coding:utf8 -*-
# @TIME     : 2019/3/29 9:31
# @Author   : SuHao
# @File     : feature_extraction.py

import scipy.signal as signal
import scipy.fftpack as fftpack
import numpy as np
import librosa as lib
import pyAudioAnalysis.audioFeatureExtraction as pyaudio
import frft
from librosa import filters
import visualization as visual

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
    if pic is not None:
        visual.stft_specgram(f, t, zxx, picname=pic + '_stft')
    return f, t, zxx


def fft_singleside(x, fs, pic=None):
    '''
    :param: x:输入信号
    :return: 偶数点fft的单边谱。
    '''
    X = fftpack.fft(x)
    X = X[0: len(X)//2 + 1]
    if pic is not None:
        visual.picplot(
            x=np.linspace(0, np.pi, len(X), endpoint=True),
            y=np.abs(X),
            title='FFT单边谱',
            xlabel='freq',
            ylabel='mag',
            pic=pic + 'fft_single_spectral')
    f = np.linspace(0, np.pi, len(X), endpoint=True)*fs/np.pi/2
    return X, f


def zero_crossing_rate(frames, pic=None):
    # 特征列向量
    # array也是可迭代的，按照行迭代
    # 使用map（func, array)相当于for循环遍历行
    # frame每一列是一帧
    feature = np.array(list(map(pyaudio.stZCR, frames.T)))
    if feature.shape[0] == frames.shape[1]:
        if pic is not None:
            visual.picplot(
                x=np.arange(0, len(feature)),
                y=feature,
                title='zero_crossing_rate',
                xlabel='time',
                ylabel='zero_crossing_rate',
                pic=pic + '_zero_crossing_rate')
        return np.array([feature])
    else:
        raise Exception("zero_crossing_rate wrong")


def energy(frames, pic=None):
    feature = np.array(list(map(pyaudio.stEnergy, frames.T)))
    if feature.shape[0] == frames.shape[1]:
        if pic is not None:
            visual.picplot(
                x=np.arange(0, len(feature)),
                y=feature,
                title='energy',
                xlabel='time',
                ylabel='energy',
                pic=pic + 'energy')
        return np.array([feature])
    else:
        raise Exception("energy wrong")


def entropy_of_energy(frames, n_short_blocks=10, pic=None):
    feature = np.array(list(map(pyaudio.stEnergyEntropy, frames.T,
                                n_short_blocks * np.ones((frames.shape[1], )).astype('int'))))
    if feature.shape[0] == frames.shape[1]:
        if pic is not None:
            visual.picplot(
                x=np.arange(0, len(feature)),
                y=feature,
                title='entropy_of_energy',
                xlabel='time',
                ylabel='entropy_of_energy',
                pic=pic + '_entropy_of_energy')
        return np.array([feature])
    else:
        raise Exception("entropy_of_energy wrong")


def spectral_centroid_spread(X, fs, pic=None):
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
        if pic is not None:
            visual.picplot(
                x=np.arange(0, len(feature[:, 0])),
                y=feature[:, 0],
                title='spectral_centroid',
                xlabel='time',
                ylabel='spectral_centroid',
                pic=pic + '_spectral_centroid')
            visual.picplot(
                x=np.arange(0, len(feature[:, 1])),
                y=feature[:, 1],
                title='spectral_spread',
                xlabel='time',
                ylabel='spectral_spread',
                pic=pic + '_spectral_spread')
        return np.array([feature[:, 0]]), np.array([feature[:, 1]])
    else:
        raise Exception("spectral_centroid_spread wrong")


def spectral_entropy(X, n_short_blocks=10, pic=None):
    y = np.abs(X)
    feature = np.array(list(map(pyaudio.stSpectralEntropy, y.T,
                                n_short_blocks * np.ones((y.shape[1],)).astype('int'))))
    if feature.shape[0] == y.shape[1]:
        if pic is not None:
            visual.picplot(
                x=np.arange(0, len(feature)),
                y=feature,
                title='spectral_entropy',
                xlabel='time',
                ylabel='spectral_entropy',
                pic=pic + '_spectral_entropy')
        return np.array([feature])
    else:
        raise Exception("spectral_entropy wrong")


def spectral_flux(X, pic=None):
    y1 = np.abs(X)
    y2 = np.concatenate([np.zeros((X.shape[0], 1)), y1[:, 1:]], axis=1)
    feature = np.array(
        list(map(pyaudio.stSpectralFlux, y1.T, y2.T)))
    if feature.shape[0] == y1.shape[1]:
        if pic is not None:
            visual.picplot(
                x=np.arange(0, len(feature)),
                y=feature,
                title='spectral_flux',
                xlabel='time',
                ylabel='spectral_flux',
                pic=pic + '_spectral_flux')
        return np.array([feature])
    else:
        raise Exception("spectral_flux wrong")
    pass


def spectral_rolloff(X, c, fs, pic=None):
    c = np.array(c).astype('float32')
    y = np.abs(X)
    feature = np.array(list(map(pyaudio.stSpectralRollOff, y.T,
                                c * np.ones((y.shape[1],)), fs * np.ones((y.shape[1],)))))
    if feature.shape[0] == y.shape[1]:
        if pic is not None:
            visual.picplot(
                x=np.arange(0, len(feature)),
                y=feature,
                title='spectral_rolloff',
                xlabel='time',
                ylabel='spectral_rolloff',
                pic=pic + '_spectral_rolloff')
        return np.array([feature])
    else:
        raise Exception("spectral_flux wrong")
    pass


def bandwidth(X, freq, norm=True, p=2, pic=None):
    '''
    :param X:
    :param params: sr:采样频率；n_fft：fft长度，4；hop_length：窗口移动步长；
                   X=频谱图，freq=频谱图对应的频率, centroid=None, norm=True, p=2
    :return:
    '''
    y = np.abs(X)
    feature = lib.feature.spectral_bandwidth(S=y, freq=freq, norm=norm, p=p)
    if pic is not None:
        visual.picplot(
            x=np.arange(0, feature.shape[1]),
            y=feature[0, :],
            title='bandwidth',
            xlabel='time',
            ylabel='bandwidth',
            pic=pic + '_bandwidth')
    return feature


def mfccs(X, fs, nfft, n_mels=128, n_mfcc=13, dct_type=2, pic=None):
    '''
    y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho'
    '''
    # 每一个帧有13个MFCC特征
    y = np.abs(X)**2
    S = lib.feature.melspectrogram(S=y, sr=fs, n_fft=nfft, n_mels=128)
    feature = lib.feature.mfcc(
        S=lib.power_to_db(S),
        sr=fs,
        n_mfcc=n_mfcc,
        dct_type=dct_type)
    if pic is not None:
        visual.specgram(
            X=feature,
            title='mfccs',
            xlabel='Time',
            ylabel='mfccs',
            pic=pic + '_mfccs')
    return feature


def rms(X, pic=None):
    '''
    :param X: stft频谱(复数)
    :return:
    '''
    y = np.abs(X)
    feature = lib.feature.rms(S=y)
    if pic is not None:
        visual.picplot(
            x=np.arange(0, feature.shape[1]),
            y=feature[0, :],
            title='rms',
            xlabel='time',
            ylabel='rms',
            pic=pic + '_rms')
    return feature


def chroma_vector():
    pass


def chroma_deviation():
    pass


def stfrft(frames, p=1, pic=None):
    stfrft = np.array(list(map(frft.frft, frames.T, p *
                               np.ones((frames.shape[1], )).astype('float32'))))
    if frames.shape[0] % 2 == 0:
        stfrft = stfrft[:, frames.shape[0] / 2 - 1:]
    else:
        stfrft = stfrft[:, frames.shape[0] // 2 + 1:]
    stfrft = stfrft.T
    if stfrft.shape[1] == frames.shape[1]:
        if pic is not None:
            visual.stfrft_specgram(stfrft, pic=pic+'_stfrft')
        return np.abs(stfrft)
    else:
        raise Exception("stfrft wrong")


def frft_MFCC(S, fs, n_mfcc=13, n_mels=128, dct_type=2, norm='ortho', power=2, pic=None):
    n_fft = 2 * (S.shape[0] - 1)
    # Build a Mel filter
    S = np.abs(S)**power
    mel_basis = filters.mel(
        sr=fs,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0.0,
        fmax=None,
        htk=False,
        norm=1)
    melspectrogram = np.dot(mel_basis, S)
    S_db = lib.core.power_to_db(melspectrogram)
    feature = fftpack.dct(S_db, axis=0, type=dct_type, norm=norm)[:n_mfcc]
    if pic is not None:
        visual.specgram(
            X=feature,
            title='frft_mfcc',
            xlabel='Time',
            ylabel='frft_mfccs',
            pic=pic+'_frft_mfcc')
    return feature


def fundalmental_freq(frames, fs, pic=None):
    '''
    :param frames:
    :param fs:
    :return: 谐波比和基频
    '''
    feature = np.array(list(map(pyaudio.stHarmonic, frames.T, fs * np.ones((frames.shape[1],)))))
    feature1 = feature[:, 0]
    feature2 = feature[:, 1]
    if feature1.shape[0] == frames.shape[1]:
        if pic is not None:
            visual.picplot(
                x=np.arange(0, len(feature1)),
                y=feature1,
                title='Harmonics_ratio',
                ylabel='Harmonics_ratio',
                xlabel='Time',
                pic=pic+'_Harmonics_ratio')
            visual.picplot(
                x=np.arange(0, len(feature2)),
                y=feature2,
                title='fundalmental_freq',
                ylabel='fundalmental_freq',
                xlabel='Time',
                pic=pic+'_fundalmental_freq')
            for time in range(len(feature2)):
                X, f = fft_singleside(frames[:, time], fs, pic=None)
                visual.picfftandpitch(f, np.abs(X), feature2[time],
                                      title='fftandpitch',
                                      xlabel='freq',
                                      ylabel='mag',
                                      pic=pic+'_fundalmental_freq_timeat'+str(time))
        return np.array([feature1]), np.array([feature2])
    else:
        raise Exception("fundalmental_freq")
