# -*- coding:utf8 -*-
# @TIME     : 2019/4/25 20:21
# @Author   : SuHao
# @File     : testfeatureextraction.py


import numpy as np
import librosa as lib
import preprocessing
import feature_extraction as fe
import visualization as visual
import timbral_feature as timbral
import matplotlib.pyplot as plt

# ex = '..\\..\\suhao.wav'
# ex = '..\\..\\boy_and_girl\\class1\\arctic_a0012.wav'
# ex = '..\\..\\数据集2\\pre2012\\bflute\\BassFlute.mf.C4B4.aiff'
# ex = '..\\..\\cello_and_viola\\viola\\Viola.arco.ff.sulA.A4.stereo.aiff'
ex = '..\\..\\数据集2\\post2012\\cello\\Cello.arco.ff.sulA.A5.stereo.aiff'
data, fs = lib.load(ex, sr=None, mono=True, res_type='kaiser_best')
frame_length = int(0.03*fs)
frame_lap = int(0.015*fs)
# data = data + np.random.randn(len(data))
frames = preprocessing.frame(data, frame_length, frame_lap)
f, t, stft = fe.stft(data, pic=None, fs=fs, nperseg=frame_length,
                     noverlap=frame_length-frame_lap, nfft=8192, boundary=None, padded=False)
stft = np.abs(stft)


harm_freq, harm_mag = timbral.harmonics(frames, fs, stft, f, nfft=8192, fmin=50, fmax=500,  nht=0.15)


# 绘制谐波以及频谱
i = 20
y2 = harm_freq[i]
# y2 = y2[0: 10]
visual.picfftandpitch(f, stft[:, i], y2, title='谐波提取', xlabel='freq(Hz)', ylabel='mag', pic=None)
plt.figure()
plt.plot(frames[:, i])
plt.show()

# # 绘制音色特征
# hsc = timbral.harmonic_spectral_centroid(harm_freq, harm_mag)
# hsd = timbral.harmonic_spectral_deviation(harm_mag)
# hss = timbral.harmonic_spectral_spread(hsc, harm_freq, harm_mag)
# hsv = timbral.harmonic_spectral_variation(harm_mag)
# plt.figure()
# ax1 = plt.subplot(411)
# plt.plot(hsc[0,:])
# ax1.set_ylabel('hsc')
# ax2 = plt.subplot(412)
# plt.plot(hsd[0,:])
# ax2.set_ylabel('hsd')
# ax3 = plt.subplot(413)
# ax3.set_ylabel('hss')
# plt.plot(hss[0,:])
# ax4 = plt.subplot(414)
# plt.plot(hsv[0,:])
# ax4.set_ylabel('hsd')
# plt.xlabel('frame time')
# plt.tight_layout()
# plt.show()
