from model import Model
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class Random_Forest(Model):
    def __init__(self, n_model=10):
    
        self.n_model = n_model
        # 用于保存模型的列表，训练好分类器后将对象append进去即可
        self.models = []

       #导入数据函数
    def load_data(self,name):
        if name=='iris':
            dataset = load_iris() 
            return dataset
           
        if name=='wine':
            dataset = load_wine()
            return dataset
        

    def split_data(self,dataset_name,size):
        dataset = Random_Forest.load_data(self,dataset_name)
        feature = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
        target = pd.DataFrame(data=map(lambda item: dataset.target_names[item],
                                       dataset.target), columns={'target_names'})
        # iris_split_data = pd.concat([feature, target], axis=1)
        # print(iris_split_datas)
        feature_train, feature_test, target_train, target_test = \
            train_test_split(feature, target, test_size=1-size)
        # print(feature_train,target_train)
        return feature_train, feature_test, target_train, target_test

    def train_data(self, feature=None, label=None):
        
        n = len(feature)
        for i in range(self.n_model):
            # 在训练集N随机选取n个样本  #frac=1 样本大小有放回
            randomSamples = feature.sample(n, replace=True, axis=0)
            # print(len(set(randomSamples.index.tolist()))) 大约每次选取N的2/3
            # 在所有特征M随机选取m个特征 特征无重复
            randomFeatures = randomSamples.sample(frac=1, replace=False, axis=1)
            # print(randomFeatures.columns.tolist())
            tags = self.connect(randomFeatures.columns.tolist())
            # print(tags)
            # 筛选出索引与上相同的lable
            randomLable = label.loc[randomSamples.index.tolist(),:]
            # for i,j in zip(randomFeatures.index.tolist(),randomLable.index.tolist()):
            #     print(i,j)
            model = DecisionTreeClassifier()
            model = model.fit(randomFeatures, randomLable)
            self.models.append({tags: model})
  

    def predict(self, features, target):
       
        result = []
        vote = []

        for model in self.models:
            # 获取模型的训练标签
            modelFeatures = list(model.keys())[0].split('000')[:-1]
            # print(modelFeatures)
            # 提取模型相对应标签数据
            feature = features[modelFeatures]
            # print(feature)
            # 基分类器进行预测
            r = list(model.values())[0].predict(feature)
            vote.append(r)
        # 将数组转换为矩阵 10行45列
        vote = np.array(vote)
        # print(vote.shape) # print(vote)

        for i in range(len(features)):
            # 对每棵树的投票结果进行排序选取最大的
            v = sorted(Counter(vote[:, i]).items(),
                       key=lambda x: x[1], reverse=True)
            result.append(v[0][0])
        #print(result)
        return result
      

  

    def connect(self, ls):
        s = ''
        for i in ls:
            s += i + '000'
        return s


    def test(self,dataset_name,size):

        Bcf = Random_Forest()
        featureAndTarget = Bcf.split_data(dataset_name,size)
        Bcf.train_data(featureAndTarget[0],featureAndTarget[2])
        res = Bcf.predict(features=featureAndTarget[1], target=featureAndTarget[3]['target_names'])
        right = 0
        for i, j in zip(featureAndTarget[3]['target_names'], res):
          if i == j:
              right += 1
           #print(i + '\t' + j)
        print('准确率Accuracy为' + str(right / len(res) * 100) + "%")


