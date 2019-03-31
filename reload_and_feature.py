# -*- coding:utf8 -*-
# @TIME     : 2019/3/30 21:58
# @Author   : SuHao
# @File     : reload_and_feature.py


import numpy as np
import csv
import feature_extraction as fea
import os


saveprojectpath = '..\\仿真结果\\boy_and_girl_without_pre_amphasis_hilbert'
savedata = saveprojectpath + '\\data'
savepic = saveprojectpath + '\\pic'
savetestdata = savedata + '\\' + 'test_data'
savepreprocess = savedata + '\\' + 'preprocessing_result.csv'
savetrainfeature = savedata + '\\' + 'feature_result_train.csv'
savetestfeature = savedata + '\\' + 'test_data' + '\\' + 'feature_result_test'
path = '..\\boy_and_girl'  # 数据集路径
downsample_rate = 8000
frame_length = int(0.02 * downsample_rate)  # 20ms
frame_overlap = frame_length // 2
test_rate = 0.3   # 测试数据所占的比例


children = os.listdir(savetestdata)
if len(children):
    for child in children:                # 为了防止多次测试导致测试样本数据文件夹里的文件重复，因此先删除
        os.remove(savetestdata + '\\' + child)
if not os.path.exists(savetestdata):
    os.mkdir(savetestdata)


'''
在特征提取部分就把数据集分为训练集和测试集
大体思路如下：
    1. 读取每一类标签文件下的音频文件数；
    2. 以每一类标签文件下的音频文件数为上限产生指定比例数量的随机数，对应的文件序号即为测试数据，将其放入一个列表。
        这个列表元素的个数为标签的个数，每个元素是一个按上述规定产生的随机数组
    3. 在进行特征提取时，在循环体里设置控制，将训练数据和测试书籍分开存放的两个文件。
    4. 在这个场景下，是一个集合对应一个标签。这对训练数据来说没有什么影响；但是对于测试数据，必须将每个集合分开存放。
    即使是属于同一类标签的集合，其元素也不能混合，否则会影响集合所属标签的判断。
    5. 专门创建以文件夹用来存放测试数据，以便之后读取数据更加方便。
'''
sample_num = []    # 每一类标签文件下的音频文件数
labelname = os.listdir(path)   # 获取该路径下的子文件名
for j in range(len(labelname)):
    subpath = path + '\\' + labelname[j]
    subfilename = os.listdir(subpath)  # 查看音频文件目录
    sample_num.append(len(subfilename))
test_set = []      # 每一类标签文件下的测试样本对应序号
for i in sample_num:
    test_set.append(np.random.randint(0, i, (int(i * test_rate), )))


with open(savetrainfeature, 'w', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
datafile = open(savepreprocess, encoding='utf-8')
csv_reader = csv.reader(datafile)
for row in csv_reader:
    time_series = np.array(row[2:]).astype('float32')
    ##########################################################################

    # stft的每一列一个帧对应的特征
    _, _, features = fea.stft(time_series, fs=downsample_rate,
                              nperseg=512, noverlap=128, nfft=512)
    features = np.abs(features)

    ##########################################################################
    if int(row[1]) in test_set[int(row[0])]:  # test_set[row[0]]表示对应标签的测试数据系数， row[1]是对应的文件序号
        with open(savetestfeature + '_' + row[0] + '_' + row[1] + '.csv', 'w', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            buffer = np.concatenate([features.T, int(row[0]) *
                                     np.ones((features.shape[1], 1)),int(row[1]) *
                                     np.ones((features.shape[1], 1))], axis=1)
            csv_writer.writerows(buffer)
    else:
        with open(savetrainfeature, 'a+', newline='', encoding='utf-8') as csvfile:
            csv_write = csv.writer(csvfile)
            buffer = np.concatenate([features.T, int(row[0]) *
                                     np.ones((features.shape[1], 1)), int(row[1]) *
                                     np.ones((features.shape[1], 1))], axis=1)
            # 把特征对应的标签加进来; 把同一标签下对应的文件序号加进来。
            csv_write.writerows(buffer)
    print(row[0], row[1])
