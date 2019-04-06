# -*- coding:utf8 -*-
# @TIME     : 2019/4/1 16:40
# @Author   : SuHao
# @File     : confusion_matrix.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def cm_plot(original_label, predict_label):
    plt.figure()
    cm = confusion_matrix(original_label, predict_label)   # 由原标签和预测标签生成混淆矩阵
    cax = plt.matshow(cm, cmap=plt.cm.Blues)     # 画混淆矩阵，配色风格使用cm.Blues
    plt.colorbar(cax)    # 颜色标签
    for x in range(len(cm)):
        for y in range(len(cm)):
            # plt.text(x, y, str('%f' % (cm[x, y])), va='center', ha='center')
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
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
    plt.xticks(np.arange(len(cm)) ,[str(i) for i in range(len(cm))], rotation=90)
    plt.yticks(np.arange(len(cm)) , [str(i) for i in range(len(cm))])
    plt.xlabel('True label')  # 坐标轴标签
    plt.ylabel('Predicted label')  # 坐标轴标签
    plt.show()
    return plt


def cmpic(y_true, y_pred):
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
    #set the fontsize of label.
    #for label in plt.gca().xaxis.get_ticklabels():
    #    label.set_fontsize(8)
    #text portion
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        # c = cm_normalized[y_val][x_val]
        c = cm[y_val][x_val]
        if (c > 0):
            plt.text(x_val, y_val, c, color='red', fontsize=7, va='center', ha='center')
            # plt.text(x_val, y_val, "%0.2f" %(c,), color='red', fontsize=7, va='center', ha='center')
    #offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    #show confusion matrix
    plt.show()


b = np.random.randint(0,20,(100,))
a = np.random.randint(0,20,(100,))
cmpic(a,b)


