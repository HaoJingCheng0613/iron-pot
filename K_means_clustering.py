import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from model import Model


class KMeansClustering(Model):
    def __init__(self):
        super().__init__()
        self.n_clusters = None
        self.labels = None
        self.centroids = None

    # 加载数据集
    def load_data(self, name):
        if name == "iris":
            iris = load_iris()
            X = iris.data
            y = iris.target
            feature_names = iris.feature_names

        if name == "wine":
            wine = load_wine()
            X = wine.data
            y = wine.target
            feature_names = wine.feature_names
        return X, y, feature_names

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
    def silhouetteCoefficient(self, X, labels):
        silhouette_avg = silhouette_score(X, labels)
        # print("轮廓系数: {:.2f}".format(silhouette_avg))
        print_content = "轮廓系数: {:.2f}".format(silhouette_avg)
        return print_content

    # WCSS评价指标
    def compute_wcss(self, X, labels, centroids):
        wcss = 0.0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            distance = np.linalg.norm(cluster_points - centroids[i])
            wcss += distance ** 2
        # print("WCSS: {:.2f}".format(wcss))
        print_content = "WCSS: {:.2f}".format(wcss)
        return print_content

    # DBI指数评价指标
    def dbi(self, X, labels, centroids):
        dbi = 0.0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            intra_cluster_distance = self.calculate_intra_cluster_distance(cluster_points, centroids[i])
            inter_cluster_distances = []
            for j in range(self.n_clusters):
                if j != i:
                    inter_cluster_distance = self.calculate_inter_cluster_distance(cluster_points, centroids[j])
                    inter_cluster_distances.append(inter_cluster_distance)
            dbi += (intra_cluster_distance / max(inter_cluster_distances))
        dbi /= self.n_clusters
        # print("DBI指数: {:.2f}".format(dbi))
        print_content = "DBI指数: {:.2f}".format(dbi)
        return print_content

    def calculate_intra_cluster_distance(self, cluster_points, centroid):
        distance = np.linalg.norm(cluster_points - centroid, axis=1)
        return np.mean(distance)

    def calculate_inter_cluster_distance(self, cluster_points, other_centroid):
        distance = np.linalg.norm(cluster_points - other_centroid, axis=1)
        return np.mean(distance)

    # Calinski-Harabasz指数评价指标
    def ch_scores(self, X, labels, centroids):
        ch_score = self.calculate_ch_score(X, labels, centroids)
        # print("Calinski-Harabasz指数评价指标: {:.2f}".format(ch_score))
        print_content = "Calinski-Harabasz指数评价指标: {:.2f}".format(ch_score)
        return print_content

    def calculate_ch_score(self, X, labels, centroids):
        n = len(X)
        k = self.n_clusters

        # 计算类间离散度矩阵（Between-cluster dispersion matrix）
        overall_mean = np.mean(X, axis=0)
        bc_dispersion_matrix = np.zeros((X.shape[1], X.shape[1]))
        for i in range(k):
            cluster_points = X[labels == i]
            cluster_size = len(cluster_points)
            cluster_mean = np.mean(cluster_points, axis=0)
            bc_dispersion_matrix += cluster_size * np.outer(cluster_mean - overall_mean,
                                                            cluster_mean - overall_mean)

        # 计算类内离散度矩阵（Within-cluster dispersion matrix）
        wc_dispersion_matrix = np.zeros((X.shape[1], X.shape[1]))
        for i in range(k):
            cluster_points = X[labels == i]
            cluster_mean = centroids[i]
            cluster_distances = np.sum((cluster_points - cluster_mean) ** 2, axis=1)
            wc_dispersion_matrix += np.sum(cluster_distances)

        # 计算Calinski-Harabasz指数
        ch_score = np.trace(bc_dispersion_matrix) * (n - k) / (np.trace(wc_dispersion_matrix) * (k - 1))
        return ch_score

    # 可视化聚类结果
    def plot_clusters(self, X, labels, centroids, x_label, y_label):
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for i in range(len(X)):
            plt.scatter(X[i, 0], X[i, 1], color=colors[labels[i]])
        plt.scatter(centroids[:, 0], centroids[:, 1], color='k', marker='x', s=100)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title('K-means Clustering')
        plt.show()

    def test(self, dataname, n_clusters):
        # 使用K均值聚类算法
        # kmeans = KMeansClustering(self.n_clusters)
        self.n_clusters = n_clusters
        # 加载数据
        X, y, featurenames = self.load_data(dataname)

        X_scaled = self.standardize_data(X)
        self.train_data(X_scaled, y)
        labels = self.labels
        centroids = self.centroids
        # self.silhouetteCoefficient(X_scaled, labels)
        # self.compute_wcss(X_scaled, labels, centroids)
        # self.dbi(X_scaled, labels, centroids)
        # self.ch_scores(X_scaled, labels, centroids)
        print_content = self.silhouetteCoefficient(X_scaled, labels) + "\n" + self.compute_wcss(X_scaled, labels, centroids) + "\n" + self.dbi(X_scaled, labels, centroids) + "\n" + self.ch_scores(X_scaled, labels, centroids)
        print(print_content)
        

        # 可视化聚类结果
        self.plot_clusters(X_scaled[:, :2], labels, centroids, featurenames[0], featurenames[1])
        
        return print_content
