from PCA import PCAA
from K_means_clustering import KMeansClustering
from K_nearest_neighbors import K_nearest_neighbors

if __name__ == '__main__':
    # PCA降维算法
    # 1. 创建一个算法模型对象
    PCAA_shili = PCAA()
    # 2. 调用模型对象的方法
    PCAA_shili.test(test_size=4, dataname="iris", method="kfold", n_components=2)
    
    # k均值聚类算法
    # 1. 创建一个算法模型对象
    KMeansClustering_shili = KMeansClustering(n_clusters=3)
    # 2. 调用模型对象的方法
    KMeansClustering_shili.test()
    
    # K最近邻算法
    # 1. 创建一个算法模型对象
    K_nearest_neighbors_shili = K_nearest_neighbors()
    # 2. 调用模型对象的方法
    K_nearest_neighbors_shili.test(4, "iris", "kfold")






