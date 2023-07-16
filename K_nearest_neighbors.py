from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import Model
import warnings
warnings.filterwarnings("ignore")

class K_nearest_neighbors(Model):
    def __init__(self):
        super().__init__()

    # 加载数据集
    def load_data(self, name):
        if name == "iris":
            iris = load_iris()
            X = iris.data
            y = iris.target
        if name == "wine":
            wine = load_wine()
            X = wine.data
            y = wine.target
        return X, y

    # 数据标准化
    def standardize_data(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def split_data_Random(self, dataname, test_size):
        X, y = self.load_data(dataname)
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
        precisions = []
        recalls = []
        f1s = []
        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]
            kfoldknnmodel = self.train_data(X_train, y_train)
            y_pred = kfoldknnmodel.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        mean_accuracy = np.mean(accuracies)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1 = np.mean(f1s)

        return mean_accuracy, mean_precision, mean_recall, mean_f1

    def hold_out_validation(self, dataname, test_size):
        X_train, X_test, y_train, y_test = self.split_data_Random(dataname, test_size)
        holdoutmodel = self.train_data(X_train, y_train)
        y_pred = holdoutmodel.predict(X_test)
        y_pred_prob = holdoutmodel.predict_proba(X_test)[:, 1]
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        # 计算精确率、召回率和F1值
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        return accuracy, precision, recall, f1

    def test(self, test_size, dataname, method):
        if method == "holdout":
            hold_out_accuracy, hold_out_precision, hold_out_recall, hold_out_f1 = self.hold_out_validation(dataname,
                                                                                                               test_size)
            print("留出法准确率：", hold_out_accuracy)
            print("留出法精确率（Precision）：", hold_out_precision)
            print("留出法召回率（Recall）：", hold_out_recall)
            print("留出法F1值（F1-Score）：", hold_out_f1)

        if method == "kfold":
            k_fold_accuracy,k_fold_precision, k_fold_recall, k_fold_f1 = self.k_fold_cross_validation(dataname,
                                                                                                      test_size)  # 此处testsize相当于n_splits
            print("K折交叉验证法准确率：", k_fold_accuracy)
            print("K折交叉验证法精确率（Precision）：", k_fold_precision)
            print("K折交叉验证法召回率（Recall）：", k_fold_recall)
            print("K折交叉验证法F1值（F1-Score）：", k_fold_f1)
