import PCA
from PCA import PCAA
import LinearRegression
from LinearRegression import Linear_Regression
import SVM
from SVM import SVM
import K_nearest_neighbors
from K_nearest_neighbors import K_nearest_neighbors
import LogisticRegression
from LogisticRegression import Logistic
import DecisionTree
from DecisionTree import DecisionTree
import K_means_clustering
from K_means_clustering import KMeansClustering
import NaiveBayes
from NaiveBayes import NaiveBayes
from GradientBoosting import GradientBoosting
from RandomForest import Random_Forest

def if_choose(ss):
    print("客户端已将选项结果传到服务端。。。ss========="+ss)
    if ss[0] == "A":
        data_namen = "wine"
    if ss[0] == "B":
        data_namen = "iris"
    if ss[1] == "A":
        test_sizen = 0.2
    if ss[1] == "B":
        test_sizen = 0.3
    if ss[2] == "A":
        split_methodn = "kfold"
    if ss[2] == "B":
        split_methodn = "random"
    
    # ceshistr="data_namen="+data_namen+"\n"+"typeofdata_namen="+str(type(data_namen))+"\n"+"test_sizen="+str(test_sizen)+"\n"+"typeofdata_namen="+str(type(test_sizen))+"\n"+"split_methodn="+split_methodn+"\n"+"typeofdata_namen="+str(type(split_methodn))+"\n"
    # print("ceshistr======="+ceshistr)
    # return ceshistr
    if ss[4] == "A":        
        LinearRegression_test=Linear_Regression()
        print_conten1=LinearRegression_test.train_data(dataset_name=data_namen,size=1-test_sizen)
        print_content2="你选择了线性回归算法，数据集为:"+data_namen+"，"+"\n测试集比例为:"+str(test_sizen)+"\n"
        print_content=print_content2+print_conten1
        return print_content

    if ss[4] == "B":
        
        # SVM算法
        # 1. 创建一个算法模型对象
        SVM_shili = SVM()
        # 2. 调用模型对象的方法
        data, target = SVM_shili.load_data(data_namen) # 此处可以更改数据集名称
        data_train, data_test, target_train, target_test = SVM_shili.split_data_Random(data, target, test_sizen) # 此处可以更改划分测试集大小
        b,alphas=SVM_shili.train_data(data_train, data_test, target_train, target_test)
        #print(SVM_shili.test(data_train, data_test, target_train, target_test,b,alphas))
        print_conten1 = SVM_shili.test(data_train, data_test, target_train, target_test,b,alphas)
        print_content2="你选择了SVM算法，数据集为:"+data_namen+"，"+"\n测试集比例为:"+str(test_sizen)+"\n"
        print_content=print_content2+print_conten1
        return print_content

    if ss[4] == "C":
        test_sizen2 = 0
        # K最近邻算法
        # 1. 创建一个算法模型对象
        K_nearest_neighbors_shili = K_nearest_neighbors()
        # 2. 调用模型对象的方法
        # 在留出法中testsize指测试集的比例，K折交叉验证法中代指几折，如3折，4折等
        if(split_methodn=="kfold"):
            if(test_sizen==0.2):
                test_sizen2=5
            if(test_sizen==0.3):
                test_sizen2=4
            print_content1 = K_nearest_neighbors_shili.test(test_size=test_sizen2, dataname= data_namen , method= split_methodn)
            print_content2 = "你选择了K最近邻算法，数据集为:"+data_namen+"，"+"\n测试集比例为:"+str(test_sizen)+"\n" + "划分方法为:"+split_methodn+"\n"
            print_content=print_content2+print_content1
            return print_content
        else:   
            print_content1 = K_nearest_neighbors_shili.test(test_size=test_sizen, dataname= data_namen , method= split_methodn)
            print_content2 = "你选择了K最近邻算法，数据集为:"+data_namen+"，"+"\n测试集比例为:"+str(test_sizen)+"\n" + "划分方法为:"+split_methodn+"\n"
            print_content=print_content2+print_content1
            return print_content

    if ss[4] == "D":
        # 逻辑回归
        LogisticRegression_shill = Logistic()

        print_conten1=(LogisticRegression_shill.test(data_namen, split_methodn, 1-test_sizen, "acc"))
        print_content2="你选择了逻辑回归算法，数据集为:"+data_namen+"，"+"\n测试集比例为:"+str(test_sizen)+"\n" + "划分方法为:"+split_methodn+"\n"
        print_content=print_content2+print_conten1
        return print_content
    
    if ss[4] == "E":
        #DecisionTree算法
        # 1. 创建一个算法模型对象
        DecisionTree_shili = DecisionTree()
        # 2. 调用模型对象的方法
        data, target = DecisionTree_shili.load_data(data_namen)# 此处可以更改数据集名称
        data_train, data_test, target_train, target_test = DecisionTree_shili.split_data_Random(data, target, test_sizen)# 此处可以更改划分测试集大小
        DecisionTree_shili.train_data(data_train, target_train)
        print_content1=DecisionTree_shili.test(data_train, data_test, target_train, target_test)
        print_content2="你选择了决策树算法，数据集为:"+data_namen+"，"+"\n测试集比例为:"+str(test_sizen)+"\n"
        print_content=print_content2+print_content1
        return print_content

    if ss[4] == "F":
        # k均值聚类算法
        # 1. 创建一个算法模型对象
        KMeansClustering_shili = KMeansClustering()
        # 2. 调用模型对象的方法
        # 也可以改变聚类的簇的数量，比如将3改为2或者4
        print_content1=(KMeansClustering_shili.test(dataname=data_namen, n_clusters=3))
        print_content2="你选择了k均值聚类算法，数据集为:"+data_namen+"，"+"\n"
        print_content=print_content2+print_content1
        return print_content
    
    # 随机森林
    if ss[4] == "G":
        #1.创建一个算法模型对象
        RandomForest_test=Random_Forest()
        #3.1.1调用wine数据集 采取random分割方法 60%为训练集
        print_content1 = RandomForest_test.test(dataset_name=data_namen,size=1-test_sizen)
        print_content2="你选择了随机森林算法，数据集为:"+data_namen+"，"+"\n测试集比例为:"+str(test_sizen)+"\n"
        print_content=print_content2+print_content1
        return print_content



    if ss[4] == "H":
        #朴素贝叶斯算法
        # 1. 创建一个算法模型对象
        NaiveBayes_shill = NaiveBayes()
        # 2. 调用模型对象的方法
        # 2.1 iris数据集
        # 2.1.1 k折(参数size无意义，函数内部已定为6折)
        print_content1= NaiveBayes_shill.test(data_namen, split_methodn, test_sizen, "acc")
        print_content2="你选择了朴素贝叶斯算法，数据集为:"+data_namen+"，"+"\n测试集比例为:"+str(test_sizen)+"\n" + "划分方法为:"+split_methodn+"\n"
        print_content=print_content2+print_content1
        return print_content

    if ss[4] == "I":
        test_sizen2 = 0
        # PCA降维算法
        # 1. 创建一个算法模型对象
        # 在留出法中testsize指测试集的比例，K折交叉验证法中代指几折，如3折，4折等
        PCAA_shili = PCAA()
        if(split_methodn=="kfold"):
            if(test_sizen==0.2):
                test_sizen2=5
            if(test_sizen==0.3):
                test_sizen2=4
            print_content1 = PCAA_shili.test(test_size=test_sizen2, dataname=data_namen, method=split_methodn, n_components=2)
            print_content2 = "你选择了PCA降维算法，数据集为:"+data_namen+"，"+"\n测试集比例为:"+str(test_sizen)+"\n" + "划分方法为:"+split_methodn+"\n"
            print_content=print_content2+print_content1
            return print_content
        else:   
            print_content1 = PCAA_shili.test(test_size=test_sizen, dataname=data_namen, method=split_methodn, n_components=2)
            print_content2 = "你选择了PCA降维算法，数据集为:"+data_namen+"，"+"\n测试集比例为:"+str(test_sizen)+"\n" + "划分方法为:"+split_methodn+"\n"
            print_content=print_content2+print_content1
            return print_content




    if ss[4] == "J":
        #梯度增强算法
        # 1. 创建一个算法模型对象
        GradientBoosting_shill = GradientBoosting()
        # 2. 调用模型对象的方法
        # 2.1 iris数据集
         # 2.1.1 k折(参数size无意义，函数内部已定为6折)
        print_content1 = GradientBoosting_shill.test(data_namen, split_methodn, 1-test_sizen, "acc")
        print_content2="你选择了梯度增强算法，数据集为:"+data_namen+"，"+"\n测试集比例为:"+str(test_sizen)+"\n" + "划分方法为:"+split_methodn+"\n"
        print_content=print_content2+print_content1
        return print_content

# if __name__=="__main__":
#     if_choose("ABBBA")

