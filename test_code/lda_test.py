# -*- coding:utf8 -*-
# @TIME     : 2019/5/7 10:57
# @Author   : SuHao
# @File     : lda_test.py

import math
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
from scipy import stats
import visualization as visual
import random
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


train_X = np.random.randn(1000,10)
train_Y = np.random.randint(0, 5, (1000,))

lda = LinearDiscriminantAnalysis(n_components=3)
lda.fit(train_X, train_Y)
X_new = lda.fit_transform(train_X, train_Y)
plt.figure()
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=train_Y)
plt.plot(lda.explained_variance_ratio_)
plt.show()
# predictions = lda.predict(test_X)
# predict_label = predictions
# original_label = test_Y

print(lda.explained_variance_ratio_)


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], marker='o', c=train_Y)
ax.legend()
plt.show()