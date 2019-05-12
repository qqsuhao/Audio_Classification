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
import warnings
import preprocessing
import copy
from numba import jit


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
19.1-order delta of mfccs
20.2-order delta of mfccs

'''

# stft已经检查

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


# 单边频谱已检查
def fft_singleside(x, fs, n=None, pic=None):
    '''
    :param: x:输入信号
    :return: 偶数点fft的单边谱。
    '''
    X = fftpack.fft(x=x, n=n)
    X = X[0: len(X) // 2 + 1]
    if pic is not None:
        visual.picplot(
            x=np.linspace(0, np.pi, len(X), endpoint=True),
            y=np.abs(X),
            title='FFT单边谱',
            xlabel='freq',
            ylabel='mag',
            pic=pic + 'fft_single_spectral')
    f = np.linspace(0, np.pi, len(X), endpoint=True) * fs / np.pi / 2
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
    y2 = np.concatenate([np.zeros((X.shape[0], 1)), y1[:, 0:-1]], axis=1)
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


def spectral_rolloff(X, a, fs, pic=None):
    '''
    :param X: stft频谱
    :param a: 滚降系数
    :param fs: 采样率
    :param pic:
    :return:
    '''
    c = np.array(a).astype('float32')           # 必须转为float32，否则使用库函数会出错
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
    S = lib.feature.melspectrogram(S=y, sr=fs, n_fft=nfft, n_mels=n_mels)
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


def rms(S, pic=None):
    '''
    :param S: stft频谱(复数)
    :return:
    '''
    y = np.abs(S)
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


def stfrft(frames, p=1, pic=None):
    # win = signal.get_window('hamming', frames.shape[0]).reshape(frames.shape[0], 1)
    # win = np.tile(win, (1, frames.shape[1]))
    # frames_win = win * frames
    frames_win = copy.deepcopy(frames)
    stfrft = np.array(list(map(frft.frft, frames_win.T, p *
                               np.ones((frames_win.shape[1], )).astype('float32'))))
    # stfrft = stfrft[::-1, :]
    if frames.shape[0] % 2 == 0:
        stfrft = stfrft[:, int(frames_win.shape[0] / 2) - 1:]
    else:
        stfrft = stfrft[:, frames_win.shape[0] // 2 + 1:]
    stfrft = stfrft.T
    if stfrft.shape[1] == frames_win.shape[1]:
        if pic is not None:
            visual.stfrft_specgram(stfrft, pic=pic + '_stfrft')
        return np.abs(stfrft)
    else:
        raise Exception("stfrft wrong")


def frft_MFCC(S, fs, n_mfcc=13, n_mels=128, dct_type=2, norm='ortho', power=2, pic=None):
    n_fft = 2 * (S.shape[0] - 1)
    # Build a Mel filter
    y = np.abs(S)**power
    mel_basis = filters.mel(
        sr=fs,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0.0,
        fmax=None,
        htk=False,
        norm=1)
    melspectrogram = np.dot(mel_basis, y)
    S_db = lib.core.power_to_db(melspectrogram)
    feature = fftpack.dct(S_db, axis=0, type=dct_type, norm=norm)[:n_mfcc]
    if pic is not None:
        visual.specgram(X=feature, title='frft_mfcc', xlabel='Time', ylabel='frft_mfccs', pic=pic + '_frft_mfcc')
    return feature


def fundalmental_freq(frames, fs, pic=None):
    '''
    :param frames:
    :param fs:
    :return: 谐波比和基频
    '''
    feature = np.array(
        list(map(pyaudio.stHarmonic, frames.T, fs * np.ones((frames.shape[1],)))))
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
                pic=pic + '_Harmonics_ratio')
            visual.picplot(
                x=np.arange(0, len(feature2)),
                y=feature2,
                title='fundalmental_freq',
                ylabel='fundalmental_freq',
                xlabel='Time',
                pic=pic + '_fundalmental_freq')
            for time in range(len(feature2)):
                X, f = fft_singleside(frames[:, time], fs, pic=None)
                visual.picfftandpitch(
                    f,
                    np.abs(X),
                    feature2[time],
                    title='fftandpitch',
                    xlabel='freq',
                    ylabel='mag',
                    pic=pic +
                    '_fundalmental_freq_timeat' +
                    str(time))
        return np.array([feature1]), np.array([feature2])
    else:
        raise Exception("fundalmental_freq")


def chroma_stft(S, n_chroma=12, A440=440.0, ctroct=5.0, octwidth=2, base_c=True, norm=2):
    '''
    :param S:     STFT频谱
    :param n_chroma: 色度的个数
    :param A440: 频偏
    :param ctroct:
    :param octwidth: float > 0 or None [scalar]
                    `ctroct` and `octwidth` specify a dominance window -
                    a Gaussian weighting centered on `ctroct` (in octs, A0 = 27.5Hz)
                    and with a gaussian half-width of `octwidth`.
                    Set `octwidth` to `None` to use a flat weighting.
    :param base_c:  If True, the filter bank will start at 'C'.
                    If False, the filter bank will start at 'A'.
    :param norm: Normalization factor for each filter
    :return: chromagram  : np.ndarray [shape=(n_chroma, t)]
    '''
    y = np.abs(S)
    feature = lib.feature.chroma_stft(S=y, tuning=None, n_chroma=n_chroma, A440=A440, ctroct=ctroct,
                            octwidth=octwidth, norm=norm, base_c=base_c)
    return feature


#########################################################################
# Timbral Description
def log_attack_time(x, lower_ratio, upper_ratio, fs, n):
    """
    find the time period when the signal rise from lower_ratio * max(wavedata) to upper_ratio * max(wavedata)
    return the log10(time)
    refer: https://github.com/nwang57/InstrumentClassifier/blob/master/features.py
    """
    y = x
    maxvalue = np.max(np.abs(y))
    lower = lower_ratio * maxvalue
    upper = upper_ratio * maxvalue
    start = -1
    for i in range(len(y)):
        if start == -1 and np.abs(y[i]) > lower:
            start = i
        if start != -1 and np.abs(y[i]) > upper:
            attack = np.log10((i - start) / fs)
            break
    return attack * np.ones((1, n))  # 为了和帧的数目对齐


def temoporal_centroid(S, hop_length, fs):
    y = np.abs(S)
    rms = lib.feature.rms(S=y)
    rms = rms.reshape(S.shape[1], )
    tc = (hop_length/fs)*(np.sum(np.arange(0, S.shape[1], 1)*rms)/np.sum(rms))
    # refer: MPEG-7 Audio and Beyond
    return tc * np.ones((1, S.shape[1]))  # 为了和帧的数目对齐


def f0_estimate(f, S, fs, fmin=50, fmax=500, threshold=0.2):
    '''
    :param f: stft的频率值
    :param S: stft频谱
    :param fs: 采样率
    :param fmin:
    :param fmax:
    :param threshold:
    :return: f0 shape=(,t)
    '''
    y = np.abs(S)
    pitches, mag = lib.piptrack(S=y, sr=fs, fmin=fmin, fmax=fmax, threshold=threshold)
    f0 = f[np.argmax(mag, axis=0)]
    return f0


def harmonics(nfft, nht, f, S, fs, fmin=50, fmax=500, threshold=0.2):
    '''
    refer: MPEG-7 Audio and Beyond
    :param nfft: stft点数
    :param nht: 一个系数，通常取0.15
    :param f: stft的频率值
    :param S: stft频谱
    :param fs: 采样率
    :param fmin:
    :param fmax:
    :param threshold:
    :return: harm_freq, harm_mag 谐波频率和谐波幅度
    '''
    y = np.abs(S)
    f0 = f0_estimate(f, S, fs, fmin, fmax, threshold)
    co = f0*nfft/fs
    if np.max(f0) == 0:
        num_h = 3  # 为了防止后续出错，只要要有三个
    else:
        num_h = int(0.5*fs/np.max(f0)) # 每个帧的谐波数量可能不一样，这里为方便，统一为最少的谐波数量
    harm_freq = np.zeros((num_h, S.shape[1]))
    harm_mag = np.zeros((num_h, S.shape[1]))
    for i in range(S.shape[1]):
        if f0[i] == 0:
            continue
        for h in range(1, num_h+1):
            a0 = int(np.floor((h - nht) * co[i]))
            b0 = int(np.ceil((h + nht) * co[i]))
            point = a0 + np.argmax(y[a0:b0, i])
            harm_freq[h-1, i] = f[point]
            harm_mag[h-1, i] = y[point, i]
    return harm_freq, harm_mag


def harmonic_spectral_centroid(harm_freq, harm_mag):
    if np.max(harm_mag) == 0:
        return np.zeros((1, harm_mag.shape[1]))
    warnings.filterwarnings("ignore")
    hsc = (harm_freq*harm_mag).sum(axis=0) / harm_mag.sum(axis=0)
    warnings.filterwarnings("default")
    return hsc.reshape(1, harm_mag.shape[1])


def harmonic_spectral_deviation(harm_mag, flag=False):
    '''
    :param harm_mag:
    :param flag: 提取这个特征要至少有3个谐波，为了防止出错而设
    :return:
    '''
    if np.max(harm_mag) == 0:
        return np.zeros((1, harm_mag.shape[1]))
    warnings.filterwarnings("ignore")
    if flag:
        hsd = np.zeros((harm_mag.shape[1], ))
        for col in range(harm_mag.shape[1]):
            tmp = np.where(harm_mag[:, col] > 0)
            tmp = harm_mag[tmp[0][:], col]
            if len(tmp) < 3:
                hsd[col] = np.nan
                continue
            se = np.zeros((len(tmp), ))
            se[0] = 0.5*(tmp[0] + tmp[1])
            se[1: len(tmp)-1] = (tmp[0: len(tmp)-2] + tmp[1: len(tmp)-1] + tmp[2: len(tmp)]) / 3
            se[len(tmp)-1] = 0.5 * (tmp[-2] + tmp[-1])
            hsd[col] = np.abs((np.log10(tmp) - np.log10(se))).sum() / np.log10(tmp).sum()
    else:
        se = np.zeros(harm_mag.shape)
        rows = se.shape[0]
        se[0, :] = 0.5*(harm_mag[0, :] + harm_mag[1, :])
        se[1: rows-1, :] = (harm_mag[0:rows-2, :] + harm_mag[1:rows-1, :] + harm_mag[2:rows, :]) / 3
        se[rows-1, :] = 0.5*(harm_mag[-2, :] + harm_mag[-1, :])
        hsd = np.abs((np.log10(harm_mag) - np.log10(se))).sum(axis=0) / np.log10(harm_mag).sum(axis=0)
    warnings.filterwarnings("default")
    return hsd.reshape(1, harm_mag.shape[1])


def harmonic_spectral_spread(hsc, harm_freq, harm_mag):
    if np.max(harm_mag) == 0:
        return np.zeros((1, harm_mag.shape[1]))
    warnings.filterwarnings("ignore")
    hss = ((harm_freq-hsc)**2 * harm_mag**2).sum(axis=0) / (harm_mag**2).sum(axis=0) # 这里公式没写对，大错
    hss = np.sqrt(hss) / hsc
    warnings.filterwarnings("default")
    return hss.reshape(1, harm_mag.shape[1])


def harmonic_spectral_variation(harm_mag):
    if np.max(harm_mag) == 0:               # 求解pitches_mag_CDSV时可能会有这种特殊情况
        return np.zeros((1, harm_mag.shape[1]))
    warnings.filterwarnings("ignore")       # 由于计算过程中可能会出现nan，为了防止出现警告
    cols = harm_mag.shape[1]
    hsv = np.zeros((1, cols))
    a = np.sqrt((harm_mag[:, 0:cols-1]**2).sum(axis=0))
    b = np.sqrt((harm_mag[:, 1:cols]**2).sum(axis=0))
    hsv[0, 0:cols-1] = 1 - (harm_mag[:, 0:cols-1] * harm_mag[:, 1:cols]).sum(axis=0) / (a*b)
    warnings.filterwarnings("default")
    return hsv


##################################################################################################################
def pitches_mag(f, S, fs, fmin=50, fmax=22050, threshold=0.2):
    '''
    :param f: stft对应的频率
    :param S: stft频谱
    :param fs: 采样率
    :param fmin: 搜索频谱峰值的最小频率
    :param fmax: 搜索频谱峰值的最大频率
    :param threshold: 门限
    :return: 谱峰的频率和对应的幅值
    '''
    y = np.abs(S)
    # pitches, mag = lib.piptrack(S=y, sr=fs, fmin=fmin, fmax=fmax, threshold=threshold)
    a = 15
    pitches, mag = f[0:a], np.abs(S[0:a, :])
    pitches = pitches.reshape(a, 1)
    pitches = np.tile(pitches, (1, S.shape[1]))
    # peaks = np.where(mag < 0.1)
    # pitches = f[peaks[0][:]]
    # mag = mag[peaks[0][:]]
    return pitches, mag


def pitches_mag_CDSV(f, S, fs, fmin=50, fmax=22050, threshold=0.2):
    '''
    用于描述所有谱峰的特征
    :param f: stft对应的频率
    :param S: stft频谱
    :param fs: 采样率
    :param fmin: 搜索频谱峰值的最小频率
    :param fmax: 搜索频谱峰值的最大频率
    :param threshold: 门限
    :return: 4种特征
    '''
    pitches, mag = pitches_mag(f=f, S=S, fs=fs, fmin=fmin, fmax=fmax, threshold=threshold)
    if np.max(mag) == 0:
        return np.zeros((1, mag.shape[1]))
    hsc = harmonic_spectral_centroid(pitches, mag)
    hsd = harmonic_spectral_deviation(mag, flag=True)
    hss = harmonic_spectral_spread(hsc, pitches, mag)
    hsv = harmonic_spectral_variation(mag)
    return np.concatenate([hsc, hsd, hss, hsv], axis=0)


######################################################################################
def delta_features(feature, order):
    '''
    :param feature: 二维矩阵，行数为特征的个数，列数为时间数
    :param order: 阶数，支持一阶和二阶
    :return: 二维矩阵，行数为特征的个数，列数为时间数
    '''
    if feature.shape[-1] < 7:
        width = 5
    elif feature.shape[-1] < 9:
        width = 7
    else:
        width = 9
    delta_result = lib.feature.delta(data=feature, order=order, width=width)
    return delta_result

