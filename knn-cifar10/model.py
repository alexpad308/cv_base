import operator
import numpy as np

# sklearn 实现 KNN 分类算法
from sklearn.neighbors import KNeighborsClassifier


# 本地实现 KNN 算法
class KNNClassify():
    def __init__(self, k, dis='E'):
        assert dis == 'E' or dis == 'M'  # E - 欧氏距离；M - 曼哈顿距离
        self.k = k  # k近邻数
        if dis == 'E':
            self.distance = self.e_distance
        else:
            self.distance = self.m_distance

    # L1 Manhattan Distance
    def m_distance(self, i):
        return np.sum(abs((self.x_train - np.tile(self.x_test[i], (self.x_train.shape[0], 1)))), axis=1)

    # L2 Euclidean Distance
    def e_distance(self, i):
        return np.sqrt(np.sum(((self.x_train - np.tile(self.x_test[i], (self.x_train.shape[0], 1))) ** 2), axis=1))

    # train
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    # test
    def predict(self, x_test):
        self.x_test = x_test
        num_test = self.x_test.shape[0]  # 测试集样本数量
        label = []

        for i in range(num_test):
            dis = self.distance(i)  # 计算邻近距离
            nearest_k = np.argsort(dis)  # 距离由小到大排序，返回 index
            topK = nearest_k[:self.k]  # 选取前K个距离

            classCount = {}
            # 统计每种类别的个数
            for i in topK:
                classCount[self.y_train[i]] = classCount.get(self.y_train[i], 0) + 1  # 使用get()方法不会触发空值异常
            # 类别由大到小排序
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            label.append(sortedClassCount[0][0])

        return np.array(label)
