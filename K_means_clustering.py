import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from model import Model


class KMeansClustering(Model):
    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters
        self.labels = None
        self.centroids = None

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

    def train_data(self, X, y):
        self.labels, self.centroids = self.kmeans_clustering(X)

    def kmeans_clustering(self, X):
        centroids = self.initialize_centroids(X)
        while True:
            labels = self.assign_labels(X, centroids)
            new_centroids = self.update_centroids(X, labels)
            if np.array_equal(centroids, new_centroids):
                break
            centroids = new_centroids
        return labels, centroids

    def initialize_centroids(self, X):
        np.random.seed(0)
        indices = np.random.choice(range(len(X)), size=self.n_clusters, replace=False)
        centroids = X[indices]
        return centroids

    def assign_labels(self, X, centroids):
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        return labels

    def update_centroids(self, X, labels):
        new_centroids = []
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            centroid = np.mean(cluster_points, axis=0)
            new_centroids.append(centroid)
        return np.array(new_centroids)

    # 轮廓系数
    def evaluate_clusters(self, X, labels):
        silhouette_avg = silhouette_score(X, labels)
        print("轮廓系数: {:.2f}".format(silhouette_avg))

    # 可视化聚类结果
    def plot_clusters(self, X, labels, centroids):
        colors = ['r', 'g', 'b']
        for i in range(len(X)):
            plt.scatter(X[i, 0], X[i, 1], color=colors[labels[i]])
        plt.scatter(centroids[:, 0], centroids[:, 1], color='k', marker='x', s=100)
        plt.xlabel('Sepal Length')
        plt.ylabel('Sepal Width')
        plt.title('K-means Clustering')
        plt.show()

    def test(self):
        # 使用K均值聚类算法
        kmeans = KMeansClustering(self.n_clusters)
        # 加载数据
        X, y = kmeans.load_data("iris")
        X_scaled = kmeans.standardize_data(X)
        kmeans.train_data(X_scaled, y)
        labels = kmeans.labels
        centroids = kmeans.centroids
        kmeans.evaluate_clusters(X_scaled, labels)

        # 可视化聚类结果
        kmeans.plot_clusters(X_scaled[:, :2], labels, centroids)

