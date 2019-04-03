# -*- coding:utf8 -*-
# @TIME     : 2019/3/29 9:32
# @Author   : SuHao
# @File     : visualization.py

from feature_extraction import *
import matplotlib.pyplot as plt
import librosa as lib
from sklearn.metrics import confusion_matrix
import numpy as np


def stft_specgram(x, picname=None, **params):
    f, t, zxx = stft(x, **params)
    plt.figure()
    plt.pcolormesh(t, f, (np.abs(zxx)))
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    if picname is not None:
        plt.savefig(str(picname) + '.jpg')
    plt.clf()
    plt.close()
    return t, f, zxx


def confidece_plot(
        original_label,
        predict_label,
        predict_confidence,
        pic=None):
    tmp = np.array(original_label) == np.array(predict_label)
    tmp = tmp.astype('int')
    color = ['r', 'b']
    plt.figure()
    plt.ylim([0, 1.1])
    plt.xlim([-5, len(original_label) + 5])
    plt.scatter(np.arange(0, len(original_label), 1), predict_confidence, c=[
                color[x] for x in tmp], marker='*')
    plt.title('predict_confidence')
    plt.ylabel('confidence')
    plt.xlabel('number of sample')
    if pic is not None:
        plt.savefig(str(pic) + '.jpg')
    plt.show()
    plt.clf()
    plt.close()


def cm_plot(original_label, predict_label, pic=None):
    cm = confusion_matrix(original_label, predict_label)   # 由原标签和预测标签生成混淆矩阵
    plt.figure()
    plt.matshow(cm, cmap=plt.cm.Blues)     # 画混淆矩阵，配色风格使用cm.Blues
    plt.colorbar()    # 颜色标签
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(
                x, y), horizontalalignment='center', verticalalignment='center')
            # annotate主要在图形中添加注释
            # 第一个参数添加注释
            # 第二个参数是注释的内容
            # xy设置箭头尖的坐标
            # horizontalalignment水平对齐
            # verticalalignment垂直对齐
            # 其余常用参数如下：
            # xytext设置注释内容显示的起始位置
            # arrowprops 用来设置箭头
            # facecolor 设置箭头的颜色
            # headlength 箭头的头的长度
            # headwidth 箭头的宽度
            # width 箭身的宽度
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    plt.title('confusion matrix')
    plt.tight_layout()
    if pic is not None:
        plt.savefig(str(pic) + '.jpg')
    plt.show()
    plt.clf()
    plt.close()


def specgram(x, picname=None, **params):
    '''
    :param x:
    :param picname:
    :param params: x_coords=None, y_coords=None, x_axis=None, y_axis=None, sr=22050, hop_length=512,
                             fmin=None, fmax=None, bins_per_octave=12, ax=None, **kwargs
    :return:
    '''
    lib.display.specshow(x, **params)
    plt.colorbar()
    plt.title('Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    if picname is not None:
        plt.savefig(str(picname) + '.jpg')
    plt.clf()
    plt.close()
