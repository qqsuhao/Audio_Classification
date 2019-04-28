# -*- coding:utf8 -*-
# @TIME     : 2019/4/21 17:09
# @Author   : SuHao
# @File     : humanvoice.py

import numpy as np
import matplotlib.pyplot as plt
import librosa as lib



path = '..\\..\\suhao.wav'
time_series, fs = lib.load(path, sr=None, mono=True, res_type='kaiser_best')
frame_length = int(0.05 * fs)
frame_overlap = frame_length // 2
data = lib.util.normalize(time_series, norm=np.inf, axis=0, threshold=None, fill=None)

print(fs)

plt.figure()
plt.plot(data)
plt.show()
