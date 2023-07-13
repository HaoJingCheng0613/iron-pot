import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

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


class PCAA(Model):
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

    def pca(self,X,n_components):
        pca = PCA(n_components)
        # X_train, X_test, y_train, y_test = self.split_data(dataname,testsize)
        # y_train_pca = pca.fit_transform(y_train)
        X_pca = pca.fit_transform(X)

        return X_pca

    def train_data(self, X_train, y_train, n_components):
        X_train_pca = self.pca(X_train, n_components)
        model = LogisticRegression()
        pcamodel = model.fit(X_train_pca, y_train)
        return pcamodel

    def k_fold_cross_validation(self, dataname,n_splits, n_components):
        X, y = self.load_data(dataname)
        X_scaled = self.standardize_data(X)
        X_pca = self.pca(X_scaled, n_components)
        # 模型训练与评估
        model = LogisticRegression()
        scores = cross_val_score(model, X_pca, y, cv=n_splits)  # 这里使用了5折交叉验证
        accuracy = np.mean(scores)
        return accuracy

    def hold_out_validation(self, dataname, test_size, n_components):
        X_train, X_test, y_train,y_test = self.split_data(dataname, test_size)
        X_test_pca = self.pca(X_test, n_components)
        holdoutmodel = self.train_data(X_train, y_train, n_components)
        y_pred = holdoutmodel.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


    def test(self, test_size, dataname ,method, n_components):
        if method == "holdout":
            if dataname =="iris":
                hold_out_accuracy = self.hold_out_validation(dataname, test_size, n_components)
                print("留出法准确率：", hold_out_accuracy)

            #可以补充其他数据集

        if method == "kfold":
            if dataname=="iris":
                k_fold_accuracy = self.k_fold_cross_validation(dataname, test_size,n_components) #此处testsize相当于n_splits
                print("K折交叉验证法准确率：", k_fold_accuracy)



if __name__ == '__main__':
    # 1. 创建一个算法模型对象
    PCAA_shili = PCAA()
    # 2. 调用模型对象的方法
    PCAA_shili.test(test_size=4, dataname="iris", method="kfold", n_components=2)
