# -*- coding:utf8 -*-
# @TIME     : 2019/3/26 17:06
# @Author   : SuHao
# @File     : preprocessing.py

from preprocess_filter import *
from scipy import fftpack
import numpy as np
import librosa as lib


def pre_emphasis(x, mu, pic=True):
    z = x[2:] - mu * x[1:len(x) - 1]
    if pic:
        fig = plt.figure(figsize=(8,8),dpi=400)
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
        hx = fftpack.hilbert(x)
        z = np.sqrt(x ** 2 + hx ** 2)
        if pic:
            plt.figure(figsize=(8,8),dpi=300)
            ax1 = plt.subplot(211)
            ax1.plot(x, 'r')
            ax1.set_title('envelope')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('value')
            ax2 = plt.subplot(212)
            ax2.plot(z, 'b')
            ax2.set_title('original signal')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('value')
            plt.savefig('.\\picture\\希尔伯特方法求包络.jpg', dpi=300)
            plt.show()
        return x[z > limit]
    else:
        z = lp_filter(x, **params)
        if pic:
            plt.figure(figsize=(8,8),dpi=300)
            ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=2)
            ax1.plot(x, 'r')
            ax1.set_title('original signal')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('value')
            ax2 = plt.subplot2grid((4, 4), (2, 0), colspan=4, rowspan=2)
            ax2.plot(z, 'b')
            ax2.set_title('output of filter')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('value')
            plt.savefig('.\\picture\\低通滤波silence remove.jpg', dpi=300)
            plt.show()
        return x[z > limit]


def denoise_of_wave():
    pass


def frame():
    pass
