# -*- coding:utf8 -*-
# @TIME     : 2019/3/26 17:06
# @Author   : SuHao
# @File     : preprocessing.py

from preprocess_filter import *
import numpy as np
import librosa as lib
from pyAudioAnalysis import audioSegmentation as seg

'''
要这个库函数运行成功，除了需要安装官方的要求库以外，还需要先卸载libmagic
再安装pip install python-magic-bin，最后再把libmagic装上，
'''


def pre_emphasis(x, mu, pic=True):
    z = x[2:] - mu * x[1:len(x) - 1]
    if pic:
        fig = plt.figure()
        # ax1 = plt.subplot2grid((4,4),(0,0), colspan=4, rowspan=2)
        ax1 = plt.subplot(211)
        ax1.plot(x, 'r', lw=0.5)
        ax1.set_title('original signal')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('value')
        # ax2 = plt.subplot2grid((4,4),(2,0), colspan=4, rowspan=2)
        ax2 = plt.subplot(212)
        ax2.plot(z, 'b', lw=0.5)
        ax2.set_title('pre_emphasis')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('value')
        plt.tight_layout()
        plt.savefig('.\\picture\\预加重.jpg')
        plt.show()
    return z


def avoid_overlap(x, **params):  # 收集参数
    z = lp_filter(x, **params)  # 分配参数
    return z


def downsample(x, orig_fs, target_fs):
    return lib.resample(x, orig_fs, target_fs, res_type='scipy', fix=True)


def silence_remove(x, limit, option='filter', pic=True, **params):
    '''
    :param x: 输入信号
    :param limit: 门限
    :param option: 选择使用希尔伯特方法还是低通滤波方法
    :param pic: bool选择是否画图并保存
    :param params: 用于低通滤波器的参数
    :return:
    '''
    if option is 'hilbert':
        analytic_signal = signal.hilbert(x)
        z = np.abs(analytic_signal)
        if pic:
            plt.figure()
            ax1 = plt.subplot(311)
            ax1.plot(x, 'r')
            ax1.set_title('original signal')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('value')
            ax2 = plt.subplot(312)
            ax2.plot(z, 'b')
            ax2.set_title('envelope')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('value')
            ax3 = plt.subplot(313)
            ax3.plot(x[z > limit], 'g')
            ax3.set_title('silence_remove_hilbert')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('value')
            plt.tight_layout()
            plt.savefig('.\\picture\\silence_remove_hilbert.jpg')
            plt.show()
        return x[z > limit]
    elif option is 'SVM':
        domain = seg.silenceRemoval(x, **params)
        y = x[::]
        a = 0
        c = len(x)
        for i in domain[::-1]:
            a = i[0] * params['fs']
            b = i[1] * params['fs']
            y = np.delete(y, np.arange(b, c, 1))
            c = a
        y = np.delete(y, np.arange(0, a, 1))
        if pic:
            plt.figure()
            ax1 = plt.subplot(211)
            ax1.plot(x, 'r')
            ax1.set_title('original signal')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('value')
            ax2 = plt.subplot(212)
            ax2.plot(y, 'b')
            ax2.set_title('silence_remove_SVM')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('value')
            plt.tight_layout()
            plt.savefig('.\\picture\\silence_remove_SVM.jpg')
            plt.show()
        return y
    else:
        z = lp_filter(x, **params)
        if pic:
            plt.figure()
            ax1 = plt.subplot(311)
            ax1.plot(x, 'r')
            ax1.set_title('original signal')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('value')
            ax2 = plt.subplot(312)
            ax2.plot(z, 'b')
            ax2.set_title('output of filter')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('value')
            ax3 = plt.subplot(313)
            ax3.plot(x[z > limit], 'g')
            ax3.set_title('silence_remove_filter')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('value')
            plt.tight_layout()
            plt.savefig('.\\picture\\silence_remove_filter.jpg')
            plt.show()
        return x[z > limit]


def denoise_of_wave():
    pass


def frame():
    pass
