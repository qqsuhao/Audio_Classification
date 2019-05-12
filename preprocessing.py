# -*- coding:utf8 -*-
# @TIME     : 2019/3/26 17:06
# @Author   : SuHao
# @File     : preprocessing.py

from preprocess_filter import *
import numpy as np
import librosa as lib
from pyAudioAnalysis import audioSegmentation as seg
from copy import copy

'''
要这个库函数运行成功，除了需要安装官方的要求库以外，还需要先卸载libmagic
再安装pip install python-magic-bin，最后再把libmagic装上，
'''

# 预加重已检查完毕
def pre_emphasis(x, mu, pic=None):
    z = x[2:] - mu * x[1:len(x) - 1]
    if pic:
        plt.figure()
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
        plt.savefig(str(pic) + '.jpg')
        plt.clf()
        plt.close()
    return z


# 检查完毕
def avoid_overlap(x, **params):  # 收集参数
    '''
    防止频谱混叠
    :param x: 输入信号
    :param params: 低通滤波器相关参数
    :return: 信号响应
    '''
    z = lp_filter(x, **params)  # 分配参数
    return z


# 检查完毕
def downsample(x, orig_fs, target_fs):
    '''
    :param x: 输入信号
    :param orig_fs: 原始信号频率
    :param target_fs: 降采样目标频率
    :return: 降采样以后的信号
    '''
    # kaiser_best明显要比scipy快
    return lib.resample(
        x,
        orig_fs,
        target_fs,
        res_type='kaiser_best',
        fix=False)


# 暂未检查
def silence_remove(x, option, pic, limit, **params):
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
            plt.savefig(str(pic) + '.jpg')
            plt.clf()
            plt.close()
        return x[z > limit]
    elif option is 'HF':
        z = hilbert_filter(x=x, **params)
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
            ax3.set_title('silence_remove_hilbert_filter')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('value')
            plt.tight_layout()
            plt.savefig(str(pic) + '.jpg')
            plt.clf()
            plt.close()
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
            plt.savefig(str(pic) + '.jpg')
            # plt.show()
            plt.clf()
            plt.close()
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
            # plt.savefig(str(pic) + '.jpg')
            plt.show()
            # plt.clf()
            # plt.close()
        return x[np.abs(z) > limit]


# 检查完毕
def frame(x, frame_length, hop_length):
    # frames:np.ndarray [shape=(frame_length, N_FRAMES)] 不够一个帧长的末尾会背丢弃
    return lib.util.frame(x, frame_length, hop_length)


# 未检查
def oneset_time(x, fs, hop_length=512, backtrack=False, units='samples'):
    return lib.onset.onset_detect(y=x, sr=fs, hop_length=512, backtrack=backtrack, energy=None, units=units)


# 未检查
def smooth(x, windowLen=11):
    return seg.smoothMovingAvg(inputSignal=x, windowLen=windowLen)


# 未检查
def segment_clip(x, threshold):
    '''
    :param x: 输入信号，输入前需要先使用lib.util.normalize进行归一化
    :param threshold: 门限
    :return: 返回每一段音频开始和结束的位置
    '''
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


# 未检查
def clip_block_to_series(x, block):
    '''
    :param x: 输入信号
    :param block: 每一段音频开始和结束的位置一个二维数组
    :return: list，元素为分割好的音频信号
    '''
    buffer = []
    for i in block:
        buffer.append(x[i[0]: i[1]])
    return buffer


# 未检查
def average(y, L=1000):
    x = copy.copy(y)
    for i in range(0, len(x), L):
        x[i: i+L] = np.max(x[i: i+L])
    return x


# 未检查
def series_to_segments(x, threshold=0.2, average_len=100):
    data = lib.util.normalize(x, norm=np.inf, axis=0, threshold=None, fill=None)    # 数据归一化
    ave = average(data, average_len)                                                # 数据平滑
    block = segment_clip(ave, threshold=threshold)
    result = clip_block_to_series(data, block)
    return result
