from PCA import PCAA
from K_means_clustering import KMeansClustering
from K_nearest_neighbors import K_nearest_neighbors

if __name__ == '__main__':
    # PCA降维算法
    # 1. 创建一个算法模型对象
    # 在留出法中testsize指测试集的比例，K折交叉验证法中代指几折，如3折，4折等
    PCAA_shili = PCAA()
    # 2.1 调用模型对象的方法 1 ：
    PCAA_shili.test(test_size=4, dataname="iris", method="kfold", n_components=2)
    # 2.2 调用模型对象的方法 2 ：
    PCAA_shili.test(test_size=0.25, dataname="iris", method="random", n_components=2)
    # 2.3 调用模型对象的方法 3 ：
    PCAA_shili.test(test_size=4, dataname="wine", method="kfold", n_components=2)
    # 2.4 调用模型对象的方法 4 ：
    PCAA_shili.test(test_size=0.3, dataname="wine", method="random", n_components=2)
    
    # k均值聚类算法
    # 1. 创建一个算法模型对象
    KMeansClustering_shili = KMeansClustering()
    # 2. 调用模型对象的方法
    # 也可以改变聚类的簇的数量，比如将3改为2或者4
    KMeansClustering_shili.test(dataname="iris", n_clusters=3)
    KMeansClustering_shili.test(dataname="wine", n_clusters=3)
    
    # K最近邻算法
    # 1. 创建一个算法模型对象
    K_nearest_neighbors_shili = K_nearest_neighbors()
    # 2. 调用模型对象的方法
    # 在留出法中testsize指测试集的比例，K折交叉验证法中代指几折，如3折，4折等
    K_nearest_neighbors_shili.test(test_size=0.25, dataname="iris", method="holdout")
    K_nearest_neighbors_shili.test(test_size=0.20, dataname="wine", method="holdout")
    K_nearest_neighbors_shili.test(test_size=5, dataname="iris", method="kfold")
    K_nearest_neighbors_shili.test(test_size=4, dataname="wine", method="kfold")






