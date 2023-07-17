from PCA import PCAA
from K_means_clustering import KMeansClustering
from K_nearest_neighbors import K_nearest_neighbors
from RandomDessert import Random_Dessert
from LinearRegression import Linear_Regression
from LogisticRegression import Logistic
from NaiveBayes import NaiveBayes
from GradientBoosting import GradientBoosting

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
    K_nearest_neighbors_shili.test(test_size=0.25, dataname="iris", method="random")
    K_nearest_neighbors_shili.test(test_size=0.20, dataname="wine", method="random")
    K_nearest_neighbors_shili.test(test_size=5, dataname="iris", method="kfold")
    K_nearest_neighbors_shili.test(test_size=4, dataname="wine", method="kfold")

    #随机森林算法

    #1.创建一个算法模型对象
    RandomDessert_test=Random_Dessert()
    #3.1.1调用wine数据集 采取random分割方法 60%为训练集
    RandomDessert_test.train_data(dataset_name='wine',size=0.6)
    #3.1.2调用wine数据集 采取random分割方法 70%为训练集
    RandomDessert_test.train_data(dataset_name='wine',size=0.7)
    #3.1.3调用wine数据集 采取random分割方法 80%为训练集 
    RandomDessert_test.train_data(dataset_name='wine',size=0.8)  

    #3.2.1调用iris数据集 采取random分割方法 60%为训练集
    RandomDessert_test.train_data(dataset_name='iris',size=0.6)
    #3.1.2调用wine数据集 采取random分割方法 70%为训练集
    RandomDessert_test.train_data(dataset_name='iris',size=0.7)
    #3.1.3调用wine数据集 采取random分割方法 80%为训练集 
    RandomDessert_test.train_data(dataset_name='iris',size=0.8) 


    #线性回归算法

    #1.创建一个算法模型对象
    LinearRegression_test=Linear_Regression()
    #4.1.1调用wine数据集 采取random分割方法 60%为训练集
    LinearRegression_test.train_data(dataset_name='wine',size=0.6)
    #4.1.2调用wine数据集 采取random分割方法 70%为训练集
    LinearRegression_test.train_data(dataset_name='wine',size=0.7)
    #4.1.1调用wine数据集 采取random分割方法 80%为训练集
    LinearRegression_test.train_data(dataset_name='wine',size=0.8)

    #1.创建一个算法模型对象
    LinearRegression_test=Linear_Regression()
    #4.1.1调用iris数据集 采取random分割方法 60%为训练集
    LinearRegression_test.train_data(dataset_name='iris',size=0.6)
    #4.1.2调用iris数据集 采取random分割方法 70%为训练集
    LinearRegression_test.train_data(dataset_name='iris',size=0.7)
    #4.1.1调用iris数据集 采取random分割方法 80%为训练集
    LinearRegression_test.train_data(dataset_name='iris',size=0.8)


    #逻辑回归算法
    # 1. 创建一个算法模型对象
    LogisticRegression_shill = Logistic()
    # 2. 调用模型对象的方法
    # 2.1 iris数据集
    # 2.1.1 k折(参数size无意义，函数内部已定为6折)
    print(LogisticRegression_shill.test("iris", "kfold", 0.5, "acc"))
    print(LogisticRegression_shill.test("iris", "kfold", 0.5, "f1"))
    # 2.1.2 random
    print(LogisticRegression_shill.test("iris", "random", 0.6, "acc"))
    print(LogisticRegression_shill.test("iris", "random", 0.7, "acc"))
    print(LogisticRegression_shill.test("iris", "random", 0.8, "acc"))
    print(LogisticRegression_shill.test("iris", "random", 0.6, "f1"))
    print(LogisticRegression_shill.test("iris", "random", 0.7, "f1"))
    print(LogisticRegression_shill.test("iris", "random", 0.8, "f1"))
    #2.2 wine数据集
    # 2.2.1 k折(参数size无意义，函数内部已定为6折)
    print(LogisticRegression_shill.test("wine", "kfold", 0.5, "acc"))
    print(LogisticRegression_shill.test("wine", "kfold", 0.5, "f1"))
    # 2.2.2 random
    print(LogisticRegression_shill.test("wine", "random", 0.6, "acc"))
    print(LogisticRegression_shill.test("wine", "random", 0.7, "acc"))
    print(LogisticRegression_shill.test("wine", "random", 0.8, "acc"))
    print(LogisticRegression_shill.test("wine", "random", 0.6, "f1"))
    print(LogisticRegression_shill.test("wine", "random", 0.7, "f1"))
    print(LogisticRegression_shill.test("wine", "random", 0.8, "f1"))


    #朴素贝叶斯算法
    # 1. 创建一个算法模型对象
    NaiveBayes_shill = NaiveBayes()
    # 2. 调用模型对象的方法
    # 2.1 iris数据集
    # 2.1.1 k折(参数size无意义，函数内部已定为6折)
    print(NaiveBayes_shill.test("iris", "kfold", 0.5, "acc"))
    print(NaiveBayes_shill.test("iris", "kfold", 0.5, "f1"))
    # 2.1.2 random
    print(NaiveBayes_shill.test("iris", "random", 0.6, "acc"))
    print(NaiveBayes_shill.test("iris", "random", 0.7, "acc"))
    print(NaiveBayes_shill.test("iris", "random", 0.8, "acc"))
    print(NaiveBayes_shill.test("iris", "random", 0.6, "f1"))
    print(NaiveBayes_shill.test("iris", "random", 0.7, "f1"))
    print(NaiveBayes_shill.test("iris", "random", 0.8, "f1"))
    #2.2 wine数据集
    # 2.2.1 k折(参数size无意义，函数内部已定为6折)
    print(NaiveBayes_shill.test("wine", "kfold", 0.5, "acc"))
    print(NaiveBayes_shill.test("wine", "kfold", 0.5, "f1"))
    # 2.2.2 random
    print(NaiveBayes_shill.test("wine", "random", 0.6, "acc"))
    print(NaiveBayes_shill.test("wine", "random", 0.7, "acc"))
    print(NaiveBayes_shill.test("wine", "random", 0.8, "acc"))
    print(NaiveBayes_shill.test("wine", "random", 0.6, "f1"))
    print(NaiveBayes_shill.test("wine", "random", 0.7, "f1"))
    print(NaiveBayes_shill.test("wine", "random", 0.8, "f1"))


    #梯度增强算法
    # 1. 创建一个算法模型对象
    GradientBoosting_shill = GradientBoosting()
    # 2. 调用模型对象的方法
    # 2.1 iris数据集
    # 2.1.1 k折(参数size无意义，函数内部已定为6折)
    print(GradientBoosting_shill.test("iris", "kfold", 0.5, "acc"))
    print(GradientBoosting_shill.test("iris", "kfold", 0.5, "f1"))
    # 2.1.2 random
    print(GradientBoosting_shill.test("iris", "random", 0.6, "acc"))
    print(GradientBoosting_shill.test("iris", "random", 0.7, "acc"))
    print(GradientBoosting_shill.test("iris", "random", 0.8, "acc"))
    print(GradientBoosting_shill.test("iris", "random", 0.6, "f1"))
    print(GradientBoosting_shill.test("iris", "random", 0.7, "f1"))
    print(GradientBoosting_shill.test("iris", "random", 0.8, "f1"))
    #2.2 wine数据集
    # 2.2.1 k折(参数size无意义，函数内部已定为6折)
    print(GradientBoosting_shill.test("wine", "kfold", 0.5, "acc"))
    print(GradientBoosting_shill.test("wine", "kfold", 0.5, "f1"))
    # 2.2.2 random
    print(GradientBoosting_shill.test("wine", "random", 0.6, "acc"))
    print(GradientBoosting_shill.test("wine", "random", 0.7, "acc"))
    print(GradientBoosting_shill.test("wine", "random", 0.8, "acc"))
    print(GradientBoosting_shill.test("wine", "random", 0.6, "f1"))
    print(GradientBoosting_shill.test("wine", "random", 0.7, "f1"))
    print(GradientBoosting_shill.test("wine", "random", 0.8, "f1"))
