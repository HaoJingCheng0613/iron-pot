import numpy as np
import pandas as pd
from model import Model
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier



class Random_Dessert(Model):
    def __init__(self):
        pass

            
     #导入数据函数
    def load_data(self,name):
        if name=='iris':
            dataset = load_iris() 
            return dataset
           
        if name=='wine':
            dataset = load_wine()
            return dataset
        


    #random        
    def split_data(self,size):
        return size
   
    def train_data(self,dataset_name,size):
   
        dataset=Random_Dessert.load_data(self,dataset_name)

        X = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
        Y = pd.DataFrame(data=dataset.target)
        data = pd.concat([X, Y], axis=1)
        # 训练随机森林
        M = []  # 存储决策树模型的数组
        R = []  # 存储各决策树随机选取的特征组合
        n_trees = 8  # 设置森林中树的颗数

        # 训练多棵树
        for i in range(n_trees):
           # 随机选取样本
           sample = data.sample(frac=Random_Dessert.split_data(self,size),random_state=0)  # 对样本进行采样，目的是建造不同的树
           # 随机选取特征,随机选取k个特征组合r
           k = np.random.randint(1, sample.shape[1])  # 随机选取k个特征
           r = np.random.choice(range(sample.shape[1]), k, replace=False).tolist()  # replace=False 无放回的随机选取2个特征组合
           X = sample.iloc[:, r]

           # 选取Y
           Y = sample.iloc[:, -1]

           # 新建决策树模型
           model = DecisionTreeClassifier()
           model.fit(X, Y)

           # 存储模型
           M.append(model)  # 将决策树模型加入数组
           R.append(r)  # 将特征组合加入数组中
           print('第' + str(i) + '颗预测score=', model.score(X, Y))  # 打印每个基础模型的效果

        # 测试随机森林，将每个模型预测值的结果添加到result(DataFrame)中
        result = pd.concat([pd.DataFrame([M[i].predict(data.iloc[:, R[i]])]) for i in range(n_trees)], ignore_index=True)
        # 输出预测结果,取众数作为最终对每个样本标签的预测值
        predict = result.mode(axis=0).values[0].astype(int)
        print('预测值结果=', predict)
        # 计算准确率
        score = sum(np.where(predict == dataset.target, 1, 0)) / len(data)
        print("random分割数据的准确率为:",score)

n=Random_Dessert()
n.train_data('iris',0.8)
