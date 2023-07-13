# K最近邻
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Model:
    def __init__(self):
        pass

    def load_data(self, filepath):
        raise NotImplementedError

    def split_data(self,dataname ,test_size):
        # 这是一个抽象方法，应该在每个子类中实现
        raise NotImplementedError

    def train_data(self, X_train, y_train):
        # 这是一个抽象方法，应该在每个子类中实现
        raise NotImplementedError


class K_nearest_neighbors(Model):
    def __init__(self):
        super().__init__()

    # 加载数据集
    def load_data(self, name):
        if name == "iris":
            iris = load_iris()
            X = iris.data
            y = iris.target
            return X, y

    # 数据标准化
    def standardize_data(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def split_data(self,dataname,test_size):
        X,y = self.load_data(dataname)
        X_scaled = self.standardize_data(X)
        X_train, X_test = train_test_split(X_scaled, test_size=test_size, random_state=2023)
        y_train, y_test = train_test_split(y, test_size=test_size, random_state=2023)
        return X_train, X_test, y_train, y_test

    def train_data(self, X_train, y_train):
        knn = KNeighborsClassifier()
        knnmodel = knn.fit(X_train, y_train)
        return knnmodel

    def k_fold_cross_validation(self, dataname, n_splits):
        X, y = self.load_data(dataname)
        X_scaled = self.standardize_data(X)
        kf = KFold(n_splits=n_splits)
        accuracies = []
        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]
            kfoldknnmodel = self.train_data(X_train, y_train)
            y_pred = kfoldknnmodel.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        mean_accuracy = np.mean(accuracies)
        return mean_accuracy

    def hold_out_validation(self, dataname, test_size):
        X_train, X_test, y_train,y_test = self.split_data(dataname, test_size)
        holdoutmodel = self.train_data(X_train, y_train)
        y_pred = holdoutmodel.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


    def test(self, test_size, dataname ,method):
        if method == "holdout":
            if dataname =="iris":
                hold_out_accuracy = self.hold_out_validation(dataname, test_size)
                print("留出法准确率：", hold_out_accuracy)

            #可以补充其他数据集

        if method == "kfold":
            if dataname=="iris":
                k_fold_accuracy = self.k_fold_cross_validation(dataname, test_size) #此处testsize相当于n_splits
                print("K折交叉验证法准确率：", k_fold_accuracy)



if __name__ == '__main__':
    # 1. 创建一个算法模型对象
    K_nearest_neighbors_shili = K_nearest_neighbors()
    # 2. 调用模型对象的方法
    K_nearest_neighbors_shili.test(4, "iris", "kfold")

# # 暂时使用鸢尾花数据集
# iris = load_iris()
# X = iris.data
# y = iris.target
#
# # 标准化
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)


# def hold_out_validation(X, y, test_size):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#     knn = KNeighborsClassifier()
#     knn.fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     return accuracy
#
#
# def k_fold_cross_validation(X, y, n_splits):
#     kf = KFold(n_splits=n_splits)
#     accuracies = []
#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         knn = KNeighborsClassifier()
#         knn.fit(X_train, y_train)
#         y_pred = knn.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         accuracies.append(accuracy)
#     mean_accuracy = np.mean(accuracies)
#     return mean_accuracy


# 使用K折交叉验证法，将数据集分为5份
# n_splits = 4
# k_fold_accuracy = k_fold_cross_validation(X_scaled, y, n_splits)
# print("K折交叉验证法准确率：", k_fold_accuracy)
#
# # 使用留出法，将数据集划分为训练集和测试集
# test_size = 0.25  # 指定测试集比例为30%
# hold_out_accuracy = hold_out_validation(X_scaled, y, test_size)
# print("留出法准确率：", hold_out_accuracy)
