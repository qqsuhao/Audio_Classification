# -*- coding:utf8 -*-
# @TIME     : 2019/4/13 20:52
# @Author   : SuHao
# @File     : yaapt_test.py

import numpy as np
import matplotlib.pyplot as plt
import librosa as lib
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
import preprocessing
import feature_extraction as fea
import scipy

path = '..\\..\\boy_and_girl\\class1\\arctic_a0001.wav'
audio_data, sample_rate = lib.load(
    path, sr=None, mono=True, res_type='kaiser_best')  # 读取文件
silence_remove = preprocessing.silence_remove(
    x=audio_data,
    limit=np.max(audio_data) / 20 * 2,
    fs=sample_rate,
    option='HF',
    # pic=savepic + '\\' + 'silence_remove_hilbert_' + str(j)+'_'+str(i))
    pic=None)

signal = basic.SignalObj(silence_remove, sample_rate)

frame_time = 30.0
frame_length = int(0.03 * sample_rate)
frame_overlap = frame_length // 2 + 1
params = {'frame_length': frame_time,
          'tda_frame_length': frame_time,
          'frame_space': frame_time/2,
          'f0_min': 50.0,
          'f0_max': 1000.0,
          'fft_length': 8192,
          'bp_forder': 150,            # 带通滤波器阶数
          'bp_low': 50.0,
          'bp_hign': 1500.0,
          'nlfer_thresh1': 0.75,    # 0.75
          'nlfer_thresh2': 0.1,
          'shc_numharms': 9,             # 3
          'shc_window': 200.0,           # 40.0
          'shc_maxpeaks': 4,              # 4
          'shc_pwidth': 200.0,        # 50.0
          'shc_thresh1': 5.0,
          'shc_thresh2': 1.25,
          'f0_double': 150.0,
          'f0_half': 150.0,
          'dp5_k1': 11.0,
          'dec_factor': 1,
          'nccf_thresh1': 0.3,
          'nccf_thresh2': 0.9,
          'nccf_maaxcands': 3,
          'nccf_pwidth': 5,       # 5
          'merit_boost': 5,
          'merit_pivot': 0.20,
          'merit_extra': 0.4,
          'median_value': 7,
          'dp_w1': 0.15,
          'dp_w2': 0.5,
          'dp_w3': 100,
          'dp_w4': 0.9
          }

pitch = pYAAPT.yaapt(signal, **params)
frames = preprocessing.frame(silence_remove, frame_length, frame_overlap)
f, t, stft = fea.stft(silence_remove, pic=None, fs=sample_rate, nperseg=frame_length,
         noverlap=frame_overlap, nfft=8192, padded=True, boundary=None)
f,t,stft = scipy.signal.stft(x=silence_remove, fs=sample_rate, window='hann', nperseg=frame_length, noverlap=frame_overlap,
                  nfft=8192, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1)
print(pitch.samp_values.shape[0], frames.shape[1])
for i in range(min(pitch.samp_values.shape[0], frames.shape[1])):
    plt.figure()
    plt.subplot(211)
    X, _ = np.abs(fea.fft_singleside(x=frames[:,i], fs=sample_rate, n=8192, pic=None))
    plt.plot(np.arange(0, 8192/2+1), np.abs(stft[:,i]), 'y')
    plt.axvline(pitch.samp_interp[i], c='b')
    plt.axvline(pitch.samp_values[i], c='g')
    plt.subplot(212)
    plt.plot(np.arange(0, 8192 / 2 + 1), X, 'r')
    plt.axvline(pitch.samp_interp[i], c='b')
    plt.axvline(pitch.samp_values[i], c='g')
    plt.show()




# fig = plt.figure(1)
# plt.plot(pitch.samp_values, label='samp_values', color='blue')
# plt.plot(pitch.samp_interp, label='samp_interp', color='green')
# plt.xlabel('frames', fontsize=18)
# plt.ylabel('pitch (Hz)', fontsize=18)
# plt.legend(loc='upper right')
# axes = plt.gca()
# # axes.set_xlim([0,90])
# plt.show()