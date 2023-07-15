import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
 

class Model:
    def __init__(self):
        pass

    def load_data(self):
        raise NotImplementedError

    def split_data(self, data, test_size):
        # 这是一个抽象方法，应该在每个子类中实现
        raise NotImplementedError

    def train_data(self, X_train, y_train):
        # 这是一个抽象方法，应该在每个子类中实现
        raise NotImplementedError

class LinearRegression(Model):
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
    def split_data_random(self,name,size):
        df = pd.DataFrame(LinearRegression.load_data(self,name).data, columns=LinearRegression.load_data(self,name).feature_names)
        target = pd.DataFrame(LinearRegression.load_data(self,name).target, columns=['MEDV'])
        X = df
        y = target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-size/10, random_state=1)

        return  X_train, X_test, y_train, y_test

    def train_data(self,dataset_name,size):
        lr = LinearRegression()
        X_train,X_test, y_train, y_test=LinearRegression.split_data_random(self,dataset_name,size)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        from sklearn import metrics
        MSE = metrics.mean_squared_error(y_test, y_pred)
        RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        print('MSE:', MSE)
        print('RMSE:', RMSE)

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['font.family'] = ['sans-serif']
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['axes.unicode_minus']=False
         
        # 绘制图
        plt.figure(figsize=(15,5))
        plt.plot(range(len(y_test)), y_test, 'r', label='测试数据')
        plt.plot(range(len(y_test)), y_pred, 'b', label='预测数据')
        plt.legend()
        plt.show()
 

