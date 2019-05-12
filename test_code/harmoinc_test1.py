# -*- coding:utf8 -*-
# @TIME     : 2019/4/15 17:38
# @Author   : SuHao
# @File     : harmoinc_test1.py

import numpy as np
import librosa as lib
import scipy.signal as signal
import matplotlib.pyplot as plt
import feature_extraction as fe

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
harm_freq, harm_mag = fe.harmonics(nfft=8192, nht=0.15, f=f, S=stft, fs=fs, fmin=50, fmax=500, threshold=0.2)
LAT = fe.log_attack_time(time_series, 0.01, 0.99, fs, stft.shape[1])
TC = fe.temoporal_centroid(stft, frame_overlap, fs)
hsc = fe.harmonic_spectral_centroid(harm_freq, harm_mag)
hsd = fe.harmonic_spectral_deviation(harm_mag)
hss = fe.harmonic_spectral_spread(hsc, harm_freq, harm_mag)
hsv = fe.harmonic_spectral_variation(harm_mag)





