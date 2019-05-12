# -*- coding:utf8 -*-
# @TIME     : 2019/5/6 15:30
# @Author   : SuHao
# @File     : feature_extraction_test.py


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
ex = '..\\..\\cello_and_viola\\viola\\Viola.arco.ff.sulA.A4.stereo.aiff'
data, fs = lib.load(ex, sr=None, mono=True, res_type='kaiser_best')
frame_length = int(0.03*fs)
frame_lap = int(0.015*fs)
# data = data + np.random.randn(len(data))
frames = preprocessing.frame(data, frame_length, frame_lap)
f, t, stft = fe.stft(data, pic=None, fs=fs, nperseg=frame_length,
                     noverlap=frame_length-frame_lap, nfft=8192, boundary=None, padded=False)


pic = None
feature1 = fe.zero_crossing_rate(frames, pic=pic)
feature2 = fe.energy(frames, pic=pic)
feature3 = fe.entropy_of_energy(frames, pic=pic)
feature4, feature41 = fe.spectral_centroid_spread(stft, fs, pic=pic)
feature5 = fe.spectral_entropy(stft, pic=pic)
feature6 = fe.spectral_flux(stft, pic=pic)
feature7 = fe.spectral_rolloff(stft, 0.85, fs, pic=pic)
feature8 = fe.bandwidth(stft, f, pic=pic)
feature9 = fe.mfccs(X=stft, fs=fs, nfft=8192, n_mels=128, n_mfcc=13, pic=pic)
feature10 = fe.rms(stft, pic=pic)
feature11 = fe.stfrft(frames, p=0.95, pic=pic)
tmp = fe.stfrft(frames, p=0.95)
feature12 = fe.frft_MFCC(S=tmp, fs=fs, n_mfcc=13, n_mels=128, pic=pic)
feature19 = fe.delta_features(feature9, order=1)
feature20 = fe.delta_features(feature9, order=2)



plt.figure()
ax1 = plt.subplot(411)
plt.plot(data)
ax1.set_ylabel('original signal')
ax2 = plt.subplot(412)
plt.plot(feature1[0, :])
# ax2.set_xlabel('frame time')
ax2.set_ylabel('zero crossing rate')
ax3 = plt.subplot(413)
plt.plot(feature2[0, :])
# ax3.set_xlabel('frame time')
ax3.set_ylabel('energy')
ax4 = plt.subplot(414)
plt.plot(feature3[0, :])
# ax4.set_xlabel('frame time')
ax4.set_ylabel('entropy of energy')
plt.xlabel('frame time')
# plt.tight_layout()



plt.figure()
ax1 = plt.subplot(411)
plt.plot(feature4[0, :])
ax1.set_ylabel('spectral centroid')
ax2 = plt.subplot(412)
plt.plot(feature41[0, :])
# ax2.set_xlabel('frame time')
ax2.set_ylabel('spectral spread')
ax3 = plt.subplot(413)
plt.plot(feature5[0, :])
# ax3.set_xlabel('frame time')
ax3.set_ylabel('spectral entropy')
ax4 = plt.subplot(414)
plt.plot(feature6[0, :])
# ax4.set_xlabel('frame time')
ax4.set_ylabel('spectral flux')
plt.xlabel('frame time')
# plt.tight_layout()


plt.figure()
ax1 = plt.subplot(311)
plt.plot(feature10[0, :])
ax1.set_ylabel('rms')
ax2 = plt.subplot(312)
plt.plot(feature7[0, :])
# ax2.set_xlabel('frame time')
ax2.set_ylabel('spectral rolloff')
ax3 = plt.subplot(313)
plt.plot(feature8[0, :])
# ax3.set_xlabel('frame time')
ax3.set_ylabel('bandwidth')
plt.xlabel('frame time')


plt.figure()
ax1 = plt.subplot(411)
plt.plot(feature19[0, :],'b')
plt.plot(feature20[0, :],'g')
ax1.set_ylabel('mfcc 1')
ax2 = plt.subplot(412)
plt.plot(feature19[1, :],'b')
plt.plot(feature20[1, :],'g')
ax2.set_ylabel('mfcc 2')
ax3 = plt.subplot(413)
plt.plot(feature19[2, :],'b')
plt.plot(feature20[2, :],'g')
ax3.set_ylabel('mfcc 3')
ax4 = plt.subplot(414)
plt.plot(feature19[3, :],'b')
plt.plot(feature20[3, :],'g')
ax4.set_ylabel('mfcc 4')
plt.xlabel('frame time')
plt.legend(['1-delta-mfcc', '2-delta-mfcc'])


plt.figure()
ax1 = plt.subplot(411)
plt.plot(feature9[0, :],'r')
plt.plot(feature12[0, :],'b')
ax1.set_ylabel('mfcc and frft-mfcc 1')
ax2 = plt.subplot(412)
plt.plot(feature9[1, :],'r')
plt.plot(feature12[1, :],'b')
ax2.set_ylabel('mfcc and frft-mfcc 2')
ax3 = plt.subplot(413)
plt.plot(feature9[2, :],'r')
plt.plot(feature12[2, :],'b')
ax3.set_ylabel('mfcc and frft-mfcc 3')
ax4 = plt.subplot(414)
plt.plot(feature9[3, :],'r')
plt.plot(feature12[3, :],'b')
ax4.set_ylabel('mfcc and frft-mfcc 4')
plt.xlabel('frame time')
plt.legend(['mfcc', 'frft-mfcc'])


plt.show()


