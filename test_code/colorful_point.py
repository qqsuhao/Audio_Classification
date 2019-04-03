# -*- coding:utf8 -*-
# @TIME     : 2019/4/1 20:41
# @Author   : SuHao
# @File     : colorful_point.py

import numpy as np
import matplotlib.pyplot as plt

a = np.array([1,2,3,4,5,6])
b = a[:]

plt.scatter(a,b,c=['r','r','b','b','r','r'],s=80, marker='*')
plt.show()