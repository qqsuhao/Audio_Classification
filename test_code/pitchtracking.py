# -*- coding:utf8 -*-
# @TIME     : 2019/4/7 13:58
# @Author   : SuHao
# @File     : pitchtracking.py

import numpy as np
import librosa as lib
import scipy.signal as signal
import matplotlib.pyplot as plt

ex = '..\\..\\cello_and_viola\\cello\\Cello.arco.ff.sulA.A3.stereo.aiff'
time_series, fs = lib.load(ex, sr=None, mono=True, res_type='kaiser_best')
frame_length = int(0.03 * fs)
frame_overlap = frame_length // 2
f, t, stft = signal.stft(
    time_series,
    fs=fs,
    nperseg=frame_length,
    noverlap=frame_overlap,
    nfft=8192,
    padded=None,
    boundary=None)
pitch, mag = lib.piptrack(S=np.abs(stft), sr=fs, fmin=50, fmax=500, threshold=0.2)

for i in range(stft.shape[1]):
    plt.figure()
    plt.plot(f, np.abs(stft[:,i]))
    #for j in range(pitch.shape[0]):
    #    if pitch[j, 20] != 0:
    #        plt.axvline(pitch[j, 20], c='g')
    plt.scatter(f, (pitch[:,i]!=0).astype('float32')*mag[:,i], c='r', marker='*')
    plt.show()


