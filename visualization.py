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


def fft_specgram(f, t, S, picname=None):
    if len(S.shape) is not 2:
        return
    for i in range(S.shape[1]):
        plt.figure()
        plt.plot(f, np.abs(S[:,i]))
        plt.xlabel('frequency(Hz)')
        plt.ylabel('magnitude')
        plt.title('fft single side specgram')
        plt.tight_layout()
        if picname is not None:
            plt.savefig(str(picname) + str(t[i]) + '.jpg')
        plt.clf()
        plt.close()


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
    plt.clf()
    plt.close()


def cm_plot(y_true, y_pred, pic=None):
    def plot_confusion_matrix(cm, title='Confusion Matrix', cmap = plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    cm = confusion_matrix(y_true, y_pred)
    labels = np.arange(len(cm))
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12,8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        # c = cm_normalized[y_val][x_val]
        c = cm[y_val][x_val]
        if (c > 0):
            plt.text(x_val, y_val, c, color='red', fontsize=7, va='center', ha='center')
            # plt.text(x_val, y_val, "%0.2f" %(c,), color='red', fontsize=7, va='center', ha='center')
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plot_confusion_matrix(cm_normalized, title='confusion matrix')
    if pic is not None:
        plt.savefig(str(pic) + '.jpg', bbox_inches='tight')
    plt.clf()
    plt.close()


def picplot(x, y, title, xlabel, ylabel, pic=None):
    '''
    用于绘制特征提取的某些特征的图像。
    '''
    plt.figure()
    plt.plot(x, y)
    plt.title(str(title))
    plt.xlabel(str(xlabel))
    plt.ylabel(str(ylabel))
    plt.tight_layout()
    if pic is not None:
        plt.savefig(str(pic) + '.jpg')
    plt.clf()
    plt.close()


def stfrft_specgram(S, pic=None):
    f = np.arange(0, S.shape[0], 1)
    t = np.arange(0, S.shape[1], 1)
    plt.pcolormesh(t, f, (np.abs(S)))
    plt.colorbar()
    plt.title('STFRFT Magnitude')
    plt.ylabel('Fractional Frequency')
    plt.xlabel('Time')
    plt.tight_layout()
    if pic is not None:
        plt.savefig(str(pic) + '.jpg')
    plt.clf()
    plt.close()
