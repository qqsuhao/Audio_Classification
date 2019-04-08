# -*- coding:utf8 -*-
# @TIME     : 2019/4/7 15:41
# @Author   : SuHao
# @File     : librosa_silence_remove.py

import numpy as np
import librosa as lib
import matplotlib.pyplot as plt


path = '..\\..\\数据集2\\pre2012\\bclarinet\\BassClarinet.mf.B4Bb5.aiff'
time_series, fs = lib.load(path, sr=None, mono=True, res_type='kaiser_best')
frame_length = int(0.05 * fs)
frame_overlap = frame_length // 2
domain = lib.effects.split(time_series, top_db=42, frame_length=frame_length, hop_length=frame_overlap)
y = time_series[::]
a = 0
c = len(time_series)
for i in domain[::-1]:
    a = i[0]
    b = i[1]
    y = np.delete(y, np.arange(b, c, 1))
    c = a

plt.figure()
plt.subplot(211)
plt.plot(time_series)
plt.subplot(212)
plt.plot(y)
plt.show()