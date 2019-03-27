# -*- coding:utf8 -*-
# @TIME     : 2019/3/26 17:06
# @Author   : SuHao
# @File     : preprocessing.py


def pre_emphasis(x, mu):
    return x[2:] - mu*x[1:len(x)-1]


def denoise_of_wave():
    pass


def silence_remove():
    pass


def frame():
    pass
