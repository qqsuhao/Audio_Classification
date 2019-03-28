# -*- coding:utf8 -*-
# @TIME     : 2019/3/27 17:07
# @Author   : SuHao
# @File     : filter.py


import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt


# t = np.linspace(0, 1, 1000, False)  # 1 second
# sig = np.sin(2 * np.pi * 10 * t) + np.sin(2 *
#                                           np.pi * 20 * t)    # 构造10hz和20hz的两个信号
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.plot(t, sig)
# ax1.set_title('10 Hz and 20 Hz sinusoids')
# ax1.axis([0, 1, -2, 2])
#
# # 采样率为1000hz，带宽为15hz，输出sos
# sos = signal.butter(10, 15, 'hp', fs=1000, output='sos')
# # 将信号和通过滤波器作用，得到滤波以后的结果。在这里sos有点像冲击响应，这个函数有点像卷积的作用。
# filtered = signal.sosfilt(sos, sig)
# ax2.plot(t, filtered)
# ax2.set_title('After 15 Hz high-pass filter')
# ax2.axis([0, 1, -2, 2])
# ax2.set_xlabel('Time [seconds]')
# plt.tight_layout()
# plt.show()


b, a = signal.butter(10, 11000, 'lowpass', fs=44100, output='ba')
w, h = signal.freqz(b, a)
# 由于频域周期延拓和对称性，只需要0-pi的区间，对应频率0-fs/2
plt.plot(w / 2 / np.pi * 44100, 20 * np.log10(abs(h)))
plt.title('Butterworth filter frequency response')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.show()
