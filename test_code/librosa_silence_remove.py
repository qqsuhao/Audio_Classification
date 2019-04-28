# -*- coding:utf8 -*-
# @TIME     : 2019/4/7 15:41
# @Author   : SuHao
# @File     : librosa_silence_remove.py

from pyAudioAnalysis import audioSegmentation as seg
import numpy as np
import librosa as lib
import matplotlib.pyplot as plt
import copy



path = '..\\..\\数据集2\\pre2012\\piano\\Piano.ff.A0.aiff'
# path = '..\\..\\数据集2\\post2012\\horn\\Horn.ff.Ab4.stereo.aiff'
time_series, fs = lib.load(path, sr=None, mono=True, res_type='kaiser_best')
frame_length = int(0.05 * fs)
frame_overlap = frame_length // 2





def average(y, L=1000):
    x = copy.copy(y)
    for i in range(0, len(x), L):
        x[i: i+L] = np.max(x[i: i+L])
    return x


def segment_clip(x, threshold):
    buffer = []
    start = -1
    end = 0
    for i in range(len(x)):
        if start == -1 and np.abs(x[i]) > threshold:
            start = i
        if start != -1 and np.abs(x[i]) < threshold:
            end = i
            buffer.append(np.array([start, end]))
            start = -1
            end = 0
    return np.array(buffer)


def clip_block_to_series(x, block):
    buffer = []
    for i in block:
        buffer.append(x[i[0]: i[1]])
    return buffer


data = lib.util.normalize(time_series, norm=np.inf, axis=0, threshold=None, fill=None)
smooth = average(data, 5000)
block = segment_clip(smooth, 0.02)
result = clip_block_to_series(data, block)

# domain = lib.effects.split(time_series, top_db=42, frame_length=frame_length, hop_length=frame_overlap)
# y = time_series[::]
# a = 0
# c = len(time_series)
# for i in domain[::-1]:
#     a = i[0]
#     b = i[1]
#     y = np.delete(y, np.arange(b, c, 1))
#     c = a

plt.figure()
plt.subplot(311)
plt.plot(time_series)
plt.subplot(312)
plt.plot(data)
for i in block:
    plt.axvline(i[0])
    plt.axvline(i[1])
plt.subplot(313)
plt.plot(smooth)
for i in block:
    plt.axvline(i[0])
    plt.axvline(i[1])
plt.show()