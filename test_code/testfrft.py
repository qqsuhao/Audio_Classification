# -*- coding:utf8 -*-
# @TIME     : 2019/4/6 16:44
# @Author   : SuHao
# @File     : testfrft.py

import frft
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
'''
pip install git+ssh://git@github.com/nanaln/python_frft.git#egg=frft
'''


t=np.linspace(-4*np.pi,4*np.pi, 4000)
x=np.sin(2*np.pi*40*t + t**2)
# X=frft.FrFFT(x, np.array(np.pi).astype('float32')/2)

for i in range(101):
    X = frft.frft(x, i/100)
    plt.figure()
    plt.plot(np.abs(X))
    k = i/100
    plt.title('p='+str(k))
    plt.ylim((0, 10))
    plt.savefig('.\\frft_pic\\'+str(i)+'.jpg', dpi=300)
    plt.show()
    plt.clf()
    plt.close()
