import numpy as np
import pandas as pd
from model import Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
 

class Linear_Regression(Model):
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
        df = pd.DataFrame(Linear_Regression.load_data(self,name).data, columns=Linear_Regression.load_data(self,name).feature_names)
        target = pd.DataFrame(Linear_Regression.load_data(self,name).target, columns=['MEDV'])
        X = df
        y = target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-size/10, random_state=1)

        return  X_train, X_test, y_train, y_test

    def train_data(self,dataset_name,size):
        lr = LinearRegression()
        X_train,X_test, y_train, y_test=Linear_Regression.split_data_random(self,dataset_name,size)
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
 
