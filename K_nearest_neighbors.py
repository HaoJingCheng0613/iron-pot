from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import Model


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

    def split_data_Random(self,dataname,test_size):
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
        X_train, X_test, y_train,y_test = self.split_data_Random(dataname, test_size)
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
