# -*- coding:utf8 -*-
# @TIME     : 2019/3/29 9:32
# @Author   : SuHao
# @File     : visualization.py

from feature_extraction import *
import matplotlib.pyplot as plt
import librosa as lib
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def stft_specgram(f, t, zxx, picname=None):
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


def fft_specgram(f, t, S, picname=None):
    '''
    用于绘制fft单边时频像
    '''
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


def cm_plot(y_true, y_pred, labelname, accuracy, pic=None):
    def plot_confusion_matrix(cm, title='Confusion Matrix', labelname=labelname, accuracy=accuracy, cmap = plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title+'\n'+'accuracy: %.2f'%accuracy)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, tuple(labelname), rotation=90)
        plt.yticks(xlocations, tuple(labelname))
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
    plot_confusion_matrix(cm_normalized, title='confusion matrix', labelname=labelname, accuracy=accuracy)
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


def picfftandpitch(x,y1,y2,title, xlabel, ylabel, pic=None):
    '''
    :param x: 频率
    :param y1: fft幅值
    :param y2: 基频
    :param title:
    :param xlabel:
    :param ylabel:
    :param pic:
    :return:
    '''
    plt.figure()
    plt.plot(x, y1, 'r')
    for i in y2:
        plt.axvline(i, c='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if pic is not None:
        plt.savefig(str(pic) + '.jpg')
    # plt.show()
    # plt.clf()
    # plt.close()




def specgram(X, title, xlabel, ylabel, pic=None):
    '''
    绘制二维图
    '''
    f = np.arange(0, X.shape[0], 1)
    t = np.arange(0, X.shape[1], 1)
    plt.pcolormesh(t, f, X)
    plt.colorbar()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    if pic is not None:
        plt.savefig(str(pic) + '.jpg')
    plt.clf()
    plt.close()


def stfrft_specgram(S, pic=None):
    '''
    绘制短时分数阶傅里叶变换
    '''
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


def pca_plot(x, pca_model=None, dim=2, svd_solver='auto', pic=None):
    '''
    :param pca_model: 如果输入一个已经训练好的pca模型，那么新的数据就在这个模型上降维
    :param x: 行数为样本数，列数为属性数目
    :param dim: 目标维度
    :param svd_solver: svd分解的方法
    :param pic: 是否绘图
    :return:
    '''
    if pca_model is not None:
        newx = pca_model.transform(x)
    else:
        if dim > x.shape[1]:
            raise Exception('dim wrong')
            return
        model = PCA(n_components=dim, svd_solver=svd_solver)
        model.fit(x)
        newx = model.fit_transform(x)
    if pic is not None:
        if dim == 2:
            plt.figure()
            plt.scatter(newx[:,0], newx[:,1], '*')
            plt.title("pca dimensions to 2")
            plt.savefig(str(pic) + '_pca.jpg')
#            plt.show()
            plt.clf()
            plt.close()
        elif dim == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(newx[:,0], newx[:,1], newx[:,2], '*')
            plt.title("pca dimensions to 3")
            plt.savefig(str(pic) + '_pca.jpg')
            # plt.show()
            plt.clf()
            plt.close()
    return model


def pca_2plot(x1, x2, dim=2, svd_solver='auto', pic=None):
    '''
    :param x1:
    :param x2:
    :param dim:
    :param svd_solver:
    :param pic:
    :return:
    '''
    if dim > x1.shape[1] or dim > x2.shape[1]:
        raise Exception('dim wrong')
        return
    model = PCA(n_components=dim, svd_solver=svd_solver)
    model.fit(x1)
    newx1 = model.fit_transform(x1)
    newx2 = model.transform(x2)
    if pic is not None:
        if dim == 2:
            plt.figure()
            plt.scatter(newx1[:,0], newx1[:,1], 'r*')
            plt.scatter(newx2[:,0], newx2[:,1], 'b*')
            plt.title("pca dimensions to 2")
            plt.savefig(str(pic) + '_pca.jpg')
            # plt.show()
            plt.clf()
            plt.close()
        elif dim == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(newx1[:,0], newx1[:,1], newx1[:,2], 'r*')
            ax.scatter(newx2[:,0], newx2[:,1], newx2[:,2], 'b*')
            plt.title("pca dimensions to 3")
            plt.savefig(str(pic) + '_pca.jpg')
            # plt.show()
            plt.clf()
            plt.close()
