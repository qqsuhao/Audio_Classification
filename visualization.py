# -*- coding:utf8 -*-
# @TIME     : 2019/3/29 9:32
# @Author   : SuHao
# @File     : visualization.py

from feature_extraction import *
import matplotlib.pyplot as plt
import librosa as lib


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

def specgram(x, picname=None,**params):
    '''
    :param x:
    :param picname:
    :param params: x_coords=None, y_coords=None, x_axis=None, y_axis=None, sr=22050, hop_length=512,
                             fmin=None, fmax=None, bins_per_octave=12, ax=None, **kwargs
    :return:
    '''
    lib.display.specshow(x,**params)
    plt.colorbar()
    plt.title('Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    if picname is not None:
        plt.savefig(str(picname) + '.jpg')
    plt.clf()
    plt.close()
