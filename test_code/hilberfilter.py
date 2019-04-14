# -*- coding:utf8 -*-
# @TIME     : 2019/4/11 18:30
# @Author   : SuHao
# @File     : hilberfilter.py


import scipy.signal as signal
import numpy as np
import librosa as lib
import matplotlib.pyplot as plt
import time
from preprocess_filter import *

ex = '..\\..\\cello_and_viola\\viola\\Viola.arco.ff.sulA.A4.stereo.aiff'
time_series, fs = lib.load(ex, sr=None, mono=True, res_type='kaiser_best')


# duration = 2.0
# fs = 400.0
# samples = int(fs*duration)
# t = np.arange(samples) / fs
# time_series = signal.chirp(t, 20.0, t[-1], 100.0)
# time_series *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )


start = time.time()
env0 = hilbert_filter(time_series, fs, 201)
end = time.time()
a = end-start
print(a)

plt.figure()
plt.plot(env0)
plt.plot(time_series)
plt.show()


#sos = signal.tf2sos(b=co, a=1, pairing='nearest')
#out = signal.sosfilt(sos, time_series)
#out = signal.filtfilt(b=co, a=1, x=time_series, padlen=100)
#out = signal.convolve(co, time_series)
# envolope = np.sqrt(out**2 + time_series**2)

start = time.time()
env = np.abs(signal.hilbert(time_series))
end = time.time()
a = end-start
print(a)


plt.figure()
plt.plot(env)
plt.plot(time_series)
plt.show()


