# -*- coding:utf8 -*-
# @TIME     : 2019/4/28 17:43
# @Author   : SuHao
# @File     : silence-remove-test.py

import scipy.signal as signal
import numpy as np
import librosa as lib
import matplotlib.pyplot as plt
import time
from preprocess_filter import *
import preprocessing
import feature_extraction as fe
import time

ex = '..\\..\\数据集2\\pre2012\\bflute\\BassFlute.ff.C5B5.aiff'
# ex = '..\\..\\cello_and_viola\\viola\\Viola.arco.ff.sulA.A4.stereo.aiff'
x, fs = lib.load(ex, sr=None, mono=True, res_type='kaiser_best')
x = lib.util.normalize(x, norm=np.inf, axis=0, threshold=None, fill=None)

start = time.perf_counter()
silence_remove = preprocessing.silence_remove(
    x=x,
    limit=0.002,
    option=filter,
    pic=None,
    N=10,
    f=2000,
    fs=fs,
    plot=False)
end = time.perf_counter()
print(end-start)

start = time.perf_counter()
silence_remove2 = preprocessing.silence_remove(
    x,
    limit=None,
    option='SVM',
    pic=None,
    fs=fs,
    st_win=np.array(0.02).astype('float32'),
    st_step=np.array(0.01).astype('float32'),
    smoothWindow=0.5,
    weight=0.2,
    plot=False)
end = time.perf_counter()
print(end-start)


start = time.perf_counter()
silence_remove3 = preprocessing.silence_remove(
    x,
    limit=0.005,
    option='hilbert',
    pic=None)
end = time.perf_counter()
print(end-start)
plt.figure()
ax1 = plt.subplot(211)
ax1.plot(x, 'r')
ax1.set_title('original signal')
ax1.set_xlabel('Time')
ax1.set_ylabel('value')
ax2 = plt.subplot(212)
ax2.plot(silence_remove3, 'b')
ax2.set_title('silence_remove_Hilbert')
ax2.set_xlabel('Time')
ax2.set_ylabel('value')
plt.tight_layout()


start = time.perf_counter()
silence_remove4 = preprocessing.silence_remove(
    x=x,
    limit=0.005,
    option='HF',
    # pic=savepic + '\\' + 'silence_remove_hilbert_filter_' + str(j)+'_'+str(i))
    pic=None,
    fs=fs)
end = time.perf_counter()
print(end-start)
plt.figure()
ax1 = plt.subplot(211)
ax1.plot(x, 'r')
ax1.set_title('original signal')
ax1.set_xlabel('Time')
ax1.set_ylabel('value')
ax2 = plt.subplot(212)
ax2.plot(silence_remove4, 'b')
ax2.set_title('silence_remove_Hilbert_FIR')
ax2.set_xlabel('Time')
ax2.set_ylabel('value')
plt.tight_layout()

