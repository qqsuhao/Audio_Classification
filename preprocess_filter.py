# -*- coding:utf8 -*-
# @TIME     : 2019/3/27 15:15
# @Author   : SuHao
# @File     : preprocess_filter.py

import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt


# 低通滤波器已检查完毕
def lp_filter(x, N, f, fs, plot=False):
    '''
    :param x: 输入信号
    :param N: 巴特沃斯滤波器阶数，不能太大
    :param f: 截止频率,不是归一化频率，不过要指定采样率
    :param fs: 采样频率
    :param plot: bool。选择是否绘图且保存
    :return:
    '''
    b, a = signal.butter(N=N, Wn=f, btype='lowpass', output='ba', fs=fs)
    filtered = signal.filtfilt(b, a, x)
    if plot:
        w, h = signal.freqz(b, a)                               # 数字滤波器的频率响应
        plt.figure()
        plt.plot(w / 2 / np.pi * fs, 20 * np.log10(abs(h)))  # 由于频域周期延拓和对称性，只需要0-pi的区间，对应频率0-fs/2
        plt.title('Butterworth filter frequency response')
        plt.ylabel('Amplitude [dB]')
        plt.xlabel('freq[Hz]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.savefig(str(plot) + '幅频响应.jpg')
        # plt.show()
        plt.clf()
        plt.close()
    return filtered


# 已检查
def hilbert_filter(x, fs, order=81, pic=None):
    '''
    :param x: 输入信号
    :param fs: 信号采样频率
    :param order: 希尔伯特滤波器阶数
    :param pic: 是否绘图，bool
    :return:
    '''
    co = [2*np.sin(np.pi*n/2)**2/np.pi/n for n in range(1, order+1)]
    co1 = [2*np.sin(np.pi*n/2)**2/np.pi/n for n in range(-order, 0)]
    co = co1+[0]+ co
    # out = signal.filtfilt(b=co, a=1, x=x, padlen=int((order-1)/2))
    out = signal.convolve(x, co, mode='same', method='direct')
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


