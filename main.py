from PCA import PCAA
import K_means_clustering
from K_means_clustering import KMeansClustering
from K_nearest_neighbors import K_nearest_neighbors
from SVM import SVM
from DecisionTree import DecisionTree
from LogisticRegression import Logistic
from NaiveBayes import NaiveBayes
from GradientBoosting import GradientBoosting
from LinearRegression import Linear_Regression
import LinearRegression
import NaiveBayes
from NaiveBayes import NaiveBayes
from GradientBoosting import GradientBoosting
from RandomForest import Random_Forest

if __name__ == '__main__':
    #1.创建一个算法模型对象
    RandomForest_test=Random_Forest()
    #3.1.1调用wine数据集 采取random分割方法 60%为训练集
    RandomForest_test.test(dataset_name='wine',size=0.6)

