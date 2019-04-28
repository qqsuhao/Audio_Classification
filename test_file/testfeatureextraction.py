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
ex = '..\\..\\数据集2\\pre2012\\bflute\\BassFlute.mf.C4B4.aiff'
data, fs = lib.load(ex, sr=None, mono=True, res_type='kaiser_best')
frame_length = int(0.03*fs)
frame_lap = int(0.015*fs)
# data = data + np.random.randn(len(data))
frames = preprocessing.frame(data, frame_length, frame_lap)
stft, f = fe.mystft(frames, fs, nfft=8192)
stft = np.abs(stft)


harm_freq, harm_mag = timbral.harmonics(frames, fs, stft, f, nfft=8192, fmin=50, fmax=200,  nht=0.15)


# 绘制谐波以及频谱
# i = 25
# y2 = harm_freq[i]
# y2 = y2[0: 10]
# visual.picfftandpitch(f, stft[:, i], y2, title='谐波提取', xlabel='freq(Hz)', ylabel='mag', pic=None)
# plt.figure()
# plt.plot(frames[:, i])
# plt.show()

# 绘制音色特征
hsc = timbral.harmonic_spectral_centroid(harm_freq, harm_mag)
hsd = timbral.harmonic_spectral_deviation(harm_mag)
hss = timbral.harmonic_spectral_spread(hsc, harm_freq, harm_mag)
hsv = timbral.harmonic_spectral_variation(harm_mag)
plt.figure()
plt.subplot(411)
plt.plot(hsc[0,:])
plt.title('hsc hsd hss hsv')
plt.ylabel('mag')
plt.subplot(412)
plt.plot(hsd[0,:])
plt.ylabel('mag')
plt.subplot(413)
plt.plot(hss[0,:])
plt.ylabel('mag')
plt.subplot(414)
plt.plot(hsv[0,:])
plt.xlabel('time frame')
plt.ylabel('mag')
plt.show()
