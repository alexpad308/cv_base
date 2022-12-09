# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
# import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import KNNClassify
from config import root

batch_size = 100
pred_size = 110  # 计算测试集数目，0 代表全部数据


# visualize
def image_show(num=0):
    digit = train_loader.dataset.data[num]
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()
    print(train_loader.dataset.targets[num])


# 归一化处理：平均图
def getXmean(x_train):
    x_train = np.reshape(x_train, (x_train.shape[0], -1))  # Turn the image to 1-D
    mean_image = np.mean(x_train, axis=0)  # 求每一列均值。即求所有图片每一个像素上的平均值
    return mean_image


# 归一化处理
def centralized(x_test, mean_image):
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    x_test = x_test.astype(float)
    x_test -= mean_image  # Subtract the mean from the graph, and you get zero mean graph
    return x_test


# set MNIST dataset
train_dataset = dsets.MNIST(root=root,  # 数据集所在目录
                            train=True,  # 选择训练集
                            transform=None,  # 不使用数据预处理
                            download=True)

test_dataset = dsets.MNIST(root=root,
                           train=False,  # 选择测试集
                           transform=None,
                           download=True)

# load MNIST dataset
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True  # 数据洗牌
                          )

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True
                         )

if __name__ == '__main__':
    image_show(7)  # 可视化图片

    x_train = train_loader.dataset.data.numpy()

    # 图像归一化处理
    mean_image = getXmean(x_train)
    x_train = centralized(x_train, mean_image)

    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    y_train = train_loader.dataset.targets.numpy()

    x_test = test_loader.dataset.data.numpy()
    if pred_size:
        x_test = x_test[:pred_size]
    x_test = centralized(x_test, mean_image)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    y_test = test_loader.dataset.targets.numpy()
    if pred_size:
        y_test = y_test[:pred_size]

    num_test = y_test.shape[0]

    # 构建 KNN 分类模型
    # knn = KNNClassify(k=3, dis='E')

    # sklearn 实现 KNN分类算法
    from sklearn.neighbors import KNeighborsClassifier

    # sklearn 实现的 knn 算法优化速度更快
    # algorithm ： {‘auto’，‘ball_tree’，‘kd_tree’，‘brute’}，默认auto
    knn = KNeighborsClassifier(n_neighbors=3, p=2)  # p=2 默认代表L2距离

    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    num_correct = np.sum(y_pred == y_test)
    acc = float(num_correct) / num_test

    print('分类命中 %d / %d 个结果，正确率：%.2f%%' % (int(num_correct), num_test, acc * 100))
