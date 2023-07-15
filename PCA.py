import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from model import Model
import seaborn as sns

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

    def split_data_Random(self, dataname, test_size):
        X, y = self.load_data(dataname)
        X_scaled = self.standardize_data(X)
        X_train, X_test = train_test_split(X_scaled, test_size=test_size, random_state=2023)
        y_train, y_test = train_test_split(y, test_size=test_size, random_state=2023)
        return X_train, X_test, y_train, y_test

    def pca(self, X, n_components):
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

    def k_fold_cross_validation(self, dataname, n_splits, n_components):
        X, y = self.load_data(dataname)
        X_scaled = self.standardize_data(X)
        X_pca = self.pca(X_scaled, n_components)
        # 模型训练与评估
        model = LogisticRegression()
        scores = cross_val_score(model, X_pca, y, cv=n_splits)  # 这里使用了5折交叉验证
        y_pred = cross_val_predict(model, X_pca, y, cv=n_splits)  # 使用交叉验证预测
        precision = precision_score(y, y_pred, average="macro")
        recall = recall_score(y, y_pred, average="macro")
        f1 = f1_score(y, y_pred, average="macro")
        accuracy = np.mean(scores)
        confusion_matrix_result = confusion_matrix(y, y_pred)

        return accuracy, precision, recall, f1, confusion_matrix_result

    def hold_out_validation(self, dataname, test_size, n_components):
        X_train, X_test, y_train, y_test = self.split_data_Random(dataname, test_size)
        X_test_pca = self.pca(X_test, n_components)
        holdoutmodel = self.train_data(X_train, y_train, n_components)
        y_pred = holdoutmodel.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)
        confusion_matrix_result = confusion_matrix(y_test, y_pred)

        return accuracy, precision, recall, f1score, confusion_matrix_result

    def plt_confusion_matrix(self,confusion_matrix_result):
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()

    def test(self, test_size, dataname, method, n_components):
        if method == "holdout":
            hold_out_accuracy, hold_out_precision, hold_out_recall, hold_out_f1, hold_out_confusion_matrix_result = self.hold_out_validation(
                dataname, test_size, n_components)
            print("留出法准确率：", hold_out_accuracy)
            print("留出法精确度：", hold_out_precision)
            print("留出法召回率：", hold_out_recall)
            print("留出法F1分数：", hold_out_f1)
            self.plt_confusion_matrix(hold_out_confusion_matrix_result)


        if method == "kfold":
            k_fold_accuracy, k_fold_precision, k_fold_recall, k_fold_f1, k_fold_confusion_matrix_result = self.k_fold_cross_validation(
                dataname, test_size,
                n_components)  # 此处testsize相当于n_splits
            print("K折交叉验证法准确率：", k_fold_accuracy)
            print("K折交叉验证法精确度：", k_fold_precision)
            print("K折交叉验证法召回率：", k_fold_recall)
            print("K折交叉验证法F1分数：", k_fold_f1)
            self.plt_confusion_matrix(k_fold_confusion_matrix_result)
