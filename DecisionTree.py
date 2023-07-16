from sklearn.datasets import load_wine  # wine数据集
from sklearn.datasets import load_iris  # iris数据集
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.preprocessing import StandardScaler  # 标准差标准化
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from model import Model

class DecisionTree(Model):

    def load_data(self, filepath):
        # 在这里实现DecisionTree的数据加载逻辑
        if filepath=="wine":
            wine = load_wine()  #加载wine数据集
            data = wine['data']
            target = wine['target']
            return data,target
        if filepath=="iris":
            iris = load_iris()  # 加载wine数据集
            data = iris['data']
            target = iris['target']
            return data,target
        pass

    def split_data_Random(self, data,target, test_size):
        # 在这里实现DecisionTree的数据分割逻辑
        '''data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=test_size, random_state=125)
        return data_train, data_test, target_train, target_test'''

        num = data.shape[0]  # 样本总数
        num_test = int(num*test_size)  # 测试集样本数目
        num_train = num - num_test  # 训练集样本数目
        index = np.arange(num)  # 产生样本标号
        np.random.shuffle(index)  # 洗牌
        data_test = data[index[:num_test], :]  # 取出洗牌后前 num_test 作为测试集
        target_test = target[index[:num_test]]
        data_train = data[index[num_test:], :]  # 剩余作为训练集
        target_train = target[index[num_test:]]
        return data_train, data_test, target_train, target_test
        pass

    def train_data(self, data_train, target_train):
        # 在这里实现DecisionTree的训练逻辑
        pass


    def test(self, data_train, data_test,target_train,target_test):
        num_test = data_test.shape[0]
        # 构建决策树
        clf = tree.DecisionTreeClassifier()  # 建立决策树对象
        clf.fit(data_train, target_train)  # 决策树拟合

        # 预测
        y_test_pre = clf.predict(data_test)  # 利用拟合的决策树进行预测
        print('the predict values are', y_test_pre)  # 显示结果
        # 计算分类准确率
        acc = sum(y_test_pre == target_test) / num_test
        print('the accuracy is', acc)  # 显示预测准确率

