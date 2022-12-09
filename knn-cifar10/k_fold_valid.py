# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from knn_cifar10 import getXmean, centralized
from config import root

batch_size = 100
pred_size = 110  # 计算测试集数目，0 代表全部数据
# Cifar10 的分类
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def k_acc_show(k_choices, k_to_accuracies):
    # 绘制图像
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()


def k_fold_valid(num_folds=5, k_choices=None):
    if k_choices is None:
        k_choices = [1, 3, 5, 8, 10, 12, 15, 20]
    num_train = x_train.shape[0]

    x_train_folds = []
    y_train_folds = []

    indices = np.array_split(np.arange(num_train), indices_or_sections=num_folds)  # 将训练集分成 num_folds 份

    for i in indices:
        x_train_folds.append(x_train[i])
        y_train_folds.append(y_train[i])

    k_to_accuracies = {}

    for k in k_choices:
        acc = []
        knn = KNeighborsClassifier(n_neighbors=k, p=1)  # p=1 代表L1曼哈顿距离

        for i in range(num_folds):
            x_trn = np.concatenate(x_train_folds[:i] + x_train_folds[i + 1:], axis=0)
            y_trn = np.concatenate(y_train_folds[:i] + y_train_folds[i + 1:], axis=0)
            x_val = x_train_folds[i]
            y_val = y_train_folds[i]

            # 减少验证时间
            if pred_size:
                x_val = x_val[:pred_size]
                y_val = y_val[:pred_size]

            knn.fit(x_trn, y_trn)
            y_pred = knn.predict(x_val)

            acc.append(np.mean(y_pred == y_val))

        k_to_accuracies[k] = acc

    return k_to_accuracies


if __name__ == '__main__':

    # set Cifar10 dataset
    train_dataset = dsets.CIFAR10(root=root,  # 数据集所在目录
                                  train=True,  # 选择训练集
                                  download=True)

    test_dataset = dsets.CIFAR10(root=root,
                                 train=False,  # 选择测试集
                                 download=True)

    # load dataset
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True  # 数据洗牌
                              )

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=True
                             )

    x_train = train_loader.dataset.data
    x_train = x_train.reshape(x_train.shape[0], -1)

    # 图像归一化处理
    mean_image = getXmean(x_train)
    x_train = centralized(x_train, mean_image)

    y_train = train_loader.dataset.targets
    y_train = np.array(y_train)

    x_test = test_loader.dataset.data
    x_test = x_test.reshape(x_test.shape[0], -1)

    if pred_size:
        x_test = x_test[:pred_size]
    x_test = centralized(x_test, mean_image)
    y_test = test_loader.dataset.targets
    y_test = np.array(y_test)
    if pred_size:
        y_test = y_test[:pred_size]

    num_test = len(y_test)

    num_folds = 5  # 5折交叉验证
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20]  # 选择最优的 k 值

    k_to_accuracies = k_fold_valid(num_folds, k_choices)

    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('最近邻k = %d, 准确率 = %.2f' % (k, accuracy * 100))

    k_acc_show(k_choices, k_to_accuracies)
