from PCA import PCAA
from K_means_clustering import KMeansClustering
from K_nearest_neighbors import K_nearest_neighbors

if __name__ == '__main__':
    # PCA降维算法
    # 1. 创建一个算法模型对象
    PCAA_shili = PCAA()
    # 2.1 调用模型对象的方法 1 ：
    PCAA_shili.test(test_size=4, dataname="iris", method="kfold", n_components=2)
    # 2.2 调用模型对象的方法 2 ：
    PCAA_shili.test(test_size=4, dataname="iris", method="random", n_components=2)
    # 2.3 调用模型对象的方法 3 ：
    PCAA_shili.test(test_size=4, dataname="wine", method="kfold", n_components=2)
    # 2.4 调用模型对象的方法 4 ：
    PCAA_shili.test(test_size=4, dataname="wine", method="random", n_components=2)
    
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

    # SVM算法
    # 1. 创建一个算法模型对象
    SVM_shili = SVM()
    # 2. 调用模型对象的方法
    data, target = SVM_shili.load_data("wine") # 此处可以更改数据集名称
    data_train, data_test, target_train, target_test = SVM_shili.split_data_Random(data, target, 0.4) # 此处可以更改划分测试集大小
    b,alphas=SVM_shili.train_data(data_train, data_test, target_train, target_test)
    SVM_shili.test(data_train, data_test, target_train, target_test,b,alphas)

    
    #DecisionTree算法
    # 1. 创建一个算法模型对象
     DecisionTree_shili = DecisionTree()
    # 2. 调用模型对象的方法
    data, target = DecisionTree_shili.load_data("wine")# 此处可以更改数据集名称
    data_train, data_test, target_train, target_test = DecisionTree_shili.split_data_Random(data, target, 0.3)# 此处可以更改划分测试集大小
    DecisionTree_shili.train_data(data_train, target_train)
    DecisionTree_shili.test(data_train, data_test, target_train, target_test)


