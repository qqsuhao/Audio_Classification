# -*- coding:utf8 -*-
# @TIME     : 2019/1/18 19:46
# @Author   : SuHao
# @File     : loaddata.p
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import time

'''
minist数据集中的每个样本数据为784*1的一个数组，表示一张28*28的图片；
训练数据有60000个，测试数据有10000个
'''


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


x_train, y_train = load_mnist('.', kind='train')
x_test, y_test = load_mnist('.', kind='t10k')

fig, ax = plt.subplots(
    nrows=2,
    ncols=5,
    sharex=True,
    sharey=True, )
ax = ax.flatten()
for i in range(10):
    img = x_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

start = time.perf_counter()
neigh = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='euclidean')
neigh.fit(x_train, y_train)
Pe = neigh.score(x_test[0:10,:], y_test[0:10])
end = time.perf_counter()
print(end - start, Pe)
# neigh.predict_proba(x_test[0:10,:])
# neigh.predict(x_test[0:10,:])
