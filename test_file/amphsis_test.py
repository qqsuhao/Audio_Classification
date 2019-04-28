# -*- coding:utf8 -*-
# @TIME     : 2019/4/28 17:36
# @Author   : SuHao
# @File     : amphsis_test.py


# -*- coding:utf8 -*-
# @TIME     : 2019/4/28 11:12
# @Author   : SuHao
# @File     : downsample_test.py

import scipy.signal as signal
import numpy as np
import librosa as lib
import matplotlib.pyplot as plt
import time
from preprocess_filter import *
import preprocessing
import feature_extraction as fe

# ex = '..\\..\\数据集2\\pre2012\\bflute\\BassFlute.ff.C5B5.aiff'
ex = '..\\..\\cello_and_viola\\viola\\Viola.arco.ff.sulA.A4.stereo.aiff'
x, fs = lib.load(ex, sr=None, mono=True, res_type='kaiser_best')
downsample_rate = 22050
pre_amphasis = preprocessing.pre_emphasis(x, 0.97, pic=None)

f1, t1, S1 = fe.stft(x=x, pic=None, fs=fs, nperseg=1500, noverlap=750, nfft=8192, boundary=None, padded=False)
f2, t2, S2 = fe.stft(x=pre_amphasis, pic=None, fs=fs, nperseg=1500, noverlap=750, nfft=8192, boundary=None, padded=False)


plt.figure()
ax1 = plt.subplot(211)
ax1.set_xlabel('time')
ax1.set_ylabel('mag')
ax1.set_title('original signal')
plt.plot(x)
ax2 = plt.subplot(212)
ax2.set_xlabel('freq(Hz)')
ax2.set_ylabel('mag')
ax2.set_title('pre—amphasis')
plt.plot(pre_amphasis)
plt.tight_layout()

def stft_specgram(f, t, zxx, picname=None):
    plt.figure()
    plt.pcolormesh(t, f, (np.abs(zxx)))
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    plt.show()

stft_specgram(f1, t1, S1, picname=None)
stft_specgram(f2, t2, S2, picname=None)