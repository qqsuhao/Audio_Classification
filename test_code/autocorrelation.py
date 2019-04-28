# -*- coding:utf8 -*-
# @TIME     : 2019/4/25 14:13
# @Author   : SuHao
# @File     : autocorrelation.py

import copy
import scipy.signal as signal
import numpy as np
import librosa as lib
import matplotlib.pyplot as plt
import time
from statsmodels.tsa import stattools
import preprocessing
import feature_extraction as fe


# ex = '..\\..\\boy_and_girl\\class1\\arctic_a0012.wav'
# ex = '..\\..\\cello_and_viola\\viola\\Viola.arco.ff.sulA.A4.stereo.aiff'
time_series, fs = lib.load(ex, sr=None, mono=True, res_type='kaiser_best')


time_series = preprocessing.avoid_overlap(time_series,
                                            N=100,
                                            f=500,
                                            fs=fs,
                                            plot=False)
time_series = preprocessing.downsample(
    time_series, fs, 4410)

print(fs)
frames = preprocessing.frame(time_series, int(0.03*fs), int(0.015*fs))
for i in range(frames.shape[1]):
    acf1 = stattools.acf(frames[:,i],  nlags=100)
    fft, _ = fe.fft_singleside(frames[:,i], 4410, 8096)

    plt.figure()
    plt.subplot(211)
    plt.plot(np.abs(fft))
    plt.subplot(212)
    plt.stem(acf1)
    plt.show()


def acf_fundamental_freq(x, fmin, fmax, fs):
    y = copy.copy(x)
    y = preprocessing.avoid_overlap(y, N=100, f=fmax+100, fs=fs, plot=False)  # fmax+100为的是留出一些裕度，因为低通滤波器不理想
    # time_series = preprocessing.downsample(time_series, fs, 4410)
    nmin = int(fs / fmax)
    nmax = int(fs / fmin)
    acf = stattools.acf(y,  nlags=nmax)
    f0 = fs / np.argmax(acf[nmin:])
    return f0


def acf_fundamental_freq_frames(frames, fmin, fmax, fs):
    f0 = np.zeros((1, frames.shape[1]))
    for i in range(frames.shape[1]):
        f0[0, i] = acf_fundamental_freq(frames[:, i], fmin, fmax, fs)
    return f0
