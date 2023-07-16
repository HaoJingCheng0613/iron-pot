from model import Model
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,f1_score


class NaiveBayes(Model):

    #加载数据集
    def load_data(self, dataname):
        if dataname == "iris":
            datas = load_iris()
            x= datas.data
            y= datas.target

        elif dataname == "wine":
            datas = load_wine()
            #放入DataFrame中便于查看
            x= datas.data
            y= datas.target

        #数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(x)
        return x, y, X_scaled

    #数据分割并训练
    #k折(k=6)
    def split_data_K_Fold(self, data, ObservationIndex):
        #加载数据
        x,y,X_scaled = self.load_data(data)
        #k折分割
        kfolds = KFold(n_splits=6)
        for train_index,test_index in kfolds.split(X_scaled):
            # 准备交叉验证的数据
            x_train_fold = X_scaled[train_index]
            y_train_fold = y[train_index]
            x_test_fold = X_scaled[test_index]
            y_test_fold = y[test_index]

            # 训练
            clf = self.train_data(x_train_fold,y_train_fold)

            #评估
            tmp = self.Evaluations(clf, x_test_fold, y_test_fold, ObservationIndex)
            printer = "K折交叉验证法准确率：" + tmp
            return printer
    
    #random
    def split_data_Random(self, data, size, ObservationIndex):
        #加载数据
        x,y,X_scaled = self.load_data(data)
        #random分割
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=size, random_state=3)
        
        #训练
        clf=self.train_data(x_train, y_train)
        
        #评估
        tmp = self.Evaluations(clf, x_test, y_test, ObservationIndex)
        printer = "random" + tmp
        return printer
    
    #训练模型
    def train_data(self, X_train, y_train):
        cl = GaussianNB()
        clf = cl.fit(X_train,y_train)
        return clf

    #评估模型
    def Evaluations(self, model, x_test, y_test, ObservationIndex):
        y_pred = model.predict(x_test)
        if ObservationIndex == "acc":
            AccuracyScore = accuracy_score(y_test,y_pred)
            printer = "准确率：" +  str(AccuracyScore)
            
        elif ObservationIndex == "f1":
            F1Score = f1_score(y_test,y_pred, average='micro')
            printer = "F1分数：" +  str(F1Score)

        return printer

    #测试模型
    def test(self, dataname, method, size, ObservationIndex):
        
        if method == "kfold":
            if ObservationIndex == "acc":
                return self.split_data_K_Fold(dataname, ObservationIndex)
            elif ObservationIndex == "f1":
                return self.split_data_K_Fold(dataname, ObservationIndex)
        
        elif method == "random":
            if ObservationIndex == "acc":
                return self.split_data_Random(dataname, size, ObservationIndex)
            elif ObservationIndex == "f1":
                return self.split_data_Random(dataname, size, ObservationIndex)
   

if __name__ == '__main__':
    # 1. 创建一个算法模型对象
    a = NaiveBayes()
    # 2. 调用模型对象的方法
    print(a.test("iris", "kfold", 0.5, "f1"))
    print(a.test("wine", "random", 0.7, "f1"))
