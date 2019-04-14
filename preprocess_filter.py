# -*- coding:utf8 -*-
# @TIME     : 2019/3/27 15:15
# @Author   : SuHao
# @File     : preprocess_filter.py

import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt


def lp_filter(x, N, f, fs, plot=False):
    '''
    :param x: 输入信号
    :param N: 巴特沃斯滤波器系数
    :param f: 截止频率
    :param fs: 采样频率
    :param plot: bool。选择是否绘图且保存
    :return:
    '''
    sos = signal.butter(N, f, 'lowpass', fs=fs, output='sos')     #采样率为1000hz，带宽为15hz，输出sos
    filtered = signal.sosfilt(sos, x)             #将信号和通过滤波器作用，得到滤波以后的结果。在这里sos有点像冲击响应，这个函数有点像卷积的作用。
    b, a = signal.butter(N, f, 'lowpass', fs=fs, output='ba')
    w, h = signal.freqz(b, a)
    if plot:
        plt.figure()
        plt.plot(w / 2 / np.pi * fs, 20 * np.log10(abs(h)))  # 由于频域周期延拓和对称性，只需要0-pi的区间，对应频率0-fs/2
        plt.title('Butterworth filter frequency response')
        plt.ylabel('Amplitude [dB]')
        plt.xlabel('freq[Hz]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.savefig(str(plot) + '幅频响应.jpg')
        plt.clf()
        plt.close()
    return filtered


def hilbert_filter(x, fs, order=201, pic=None):
    co = signal.firwin(numtaps=201,
                       cutoff=[0.001*fs/2/np.pi, (np.pi-0.001)*fs/2/np.pi],
                       width=None,
                       window='hann',
                       pass_zero=False,
                       scale=True,
                       nyq=None,
                       fs=fs)
    out = signal.filtfilt(b=co, a=1, x=x, padlen=int((order-1)/2))
    envolope = np.sqrt(out**2 + x**2)
    if pic is not None:
        w, h = signal.freqz(b=co, a=1, worN=2048, whole=False, plot=None, fs=2*np.pi)
        fig, ax1 = plt.subplots()
        ax1.set_title('hilbert filter frequency response')
        ax1.plot(w, 20 * np.log10(abs(h)), 'b')
        ax1.set_ylabel('Amplitude [dB]', color='b')
        ax1.set_xlabel('Frequency [rad/sample]')
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w, angles, 'g')
        ax2.set_ylabel('Angle (radians)', color='g')
        ax2.grid()
        ax2.axis('tight')
        plt.savefig(pic + 'hilbert_filter.jpg')
        # plt.show()
        plt.clf()
        plt.close()
    return envolope


