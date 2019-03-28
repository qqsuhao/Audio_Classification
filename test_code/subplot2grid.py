# -*- coding:utf8 -*-
# @TIME     : 2019/3/28 8:14
# @Author   : SuHao
# @File     : subplot2grid.py

import matplotlib.pyplot as plt
import numpy as np

a = np.array([1,2,3,4,5,6])
plt.figure(figsize=(8,8))
ax1 = plt.subplot2grid((4,2),(0,0), colspan=2, rowspan=2)
ax1.plot(a)
ax1.set_xlabel('sdsds')
ax1.set_ylabel('xczcxc')
ax1.set_title('xvxvvcxcv')
ax2 = plt.subplot2grid((4,2),(2,0), colspan=2, rowspan=2)
ax2.plot(a**2)
ax2.set_xlabel('sdsds')
ax2.set_ylabel('xczcxc')
ax2.set_title('opopopop')
plt.show()

