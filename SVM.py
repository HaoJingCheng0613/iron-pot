from sklearn import svm
import random
from sklearn.datasets import load_wine  # wine数据集
from sklearn.datasets import load_iris  # iris数据集
from sklearn.metrics import  classification_report,precision_recall_curve
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.preprocessing import StandardScaler  # 标准差标准化
import numpy as np
import matplotlib.pyplot as plt
from model import Model


def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T  # 线性核函数,只进行内积。
    elif kTup[0] == 'rbf':  # 高斯核函数,根据高斯核函数公式进行计算
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))  # 计算高斯核K
    else:
        raise NameError('核函数无法识别')
    return K  # 返回计算的核K

class optStruct:
        def __init__(self,dataMatIn, classLabels, C, toler, kTup):
            self.X = dataMatIn  # 数据矩阵
            self.labelMat = classLabels  # 数据标签
            self.C = C  # 松弛变量
            self.tol = toler  # 容错率
            self.m = np.shape(dataMatIn)[0]  # 数据矩阵行数
            self.alphas = np.mat(np.zeros((self.m, 1)))  # 根据矩阵行数初始化alpha参数为0
            self.b = 0  # 初始化b参数为0
            # 根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
            self.eCache = np.mat(np.zeros((self.m, 2)))
            self.K = np.mat(np.zeros((self.m, self.m)))  # 初始化核K
            for i in range(self.m):  # 计算所有数据的核K
                self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJrand(i, m):
    j = i  # 选择一个不等于i的j
    while (j == i):
        j = int(random.uniform(0, m))
    return j

def selectJ(i, oS, Ei):
    maxK = -1;
    maxDeltaE = 0;
    Ej = 0  # 初始化
    oS.eCache[i] = [1, Ei]  # 根据Ei更新误差缓存
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]  # 返回误差不为0的数据的索引值
    if (len(validEcacheList)) > 1:  # 有不为0的误差
        for k in validEcacheList:  # 遍历,找到最大的Ek
            if k == i: continue  # 不计算i,浪费时间
            Ek = calcEk(oS, k)  # 计算Ek
            deltaE = abs(Ei - Ek)  # 计算|Ei-Ek|
            if (deltaE > maxDeltaE):  # 找到maxDeltaE
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxK, Ej  # 返回maxK,Ej
    else:  # 没有不为0的误差
        j = selectJrand(i, oS.m)  # 随机选择alpha_j的索引值
        Ej = calcEk(oS, j)  # 计算Ej
    return j, Ej  # j,Ej

def updateEk(oS, k):
    Ek = calcEk(oS, k)  # 计算Ek
    oS.eCache[k] = [1, Ek]  # 更新误差缓存

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def innerL(i, oS):
    # 步骤1：计算误差Ei
    Ei = calcEk(oS, i)
    # 优化alpha,设定一定的容错率。
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 使用内循环启发方式2选择alpha_j,并计算Ej
        j, Ej = selectJ(i, oS, Ei)
        # 保存更新前的aplpha值，使用深拷贝
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy();
        # 步骤2：计算上下界L和H
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        # 步骤3：计算eta
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta>=0")
            return 0
        # 步骤4：更新alpha_j
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新Ej至误差缓存
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("alpha_j变化太小")
            return 0
        # 步骤6：更新alpha_i
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 更新Ei至误差缓存
        updateEk(oS, i)
        # 步骤7：更新b_1和b_2
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[j, j]
        # 步骤8：根据b_1和b_2更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)  # 初始化数据结构
    iter = 0  # 初始化当前迭代次数
    entireSet = True;
    alphaPairsChanged = 0
    # 遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # 遍历整个数据集
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)  # 使用优化的SMO算法
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # 遍历非边界值
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]  # 遍历不在边界0和C的alpha
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:  # 遍历一次后改为非边界遍历
            entireSet = False
        elif (alphaPairsChanged == 0):  # 如果alpha没有更新,计算全样本遍历
            entireSet = True
        print("迭代次数: %d" % iter)
    return oS.b, oS.alphas  # 返回SMO算法计算的b和alphas

class SVM(Model):

    def load_data(self, filepath):
        # 在这里实现SVM的数据加载逻辑
        if filepath=="wine":
            wine = load_wine()  #加载wine数据集
            data = wine['data']
            target = wine['target']
            return data,target
        if filepath=="iris":
            iris = load_iris()  # 加载wine数据集
            data = iris['data']
            target = iris['target']
            return data,target
        pass

    def split_data_Random(self, data,target, test_size):
        # 在这里实现SVM的数据分割逻辑
        '''data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=test_size, random_state=125)
        return data_train, data_test, target_train, target_test'''
        #采用手写方式随机分割数据集
        num = data.shape[0]  # 样本总数
        num_test = int(num * test_size)  # 测试集样本数目
        num_train = num - num_test  # 训练集样本数目
        index = np.arange(num)  # 产生样本标号
        np.random.shuffle(index)  # 洗牌
        data_test = data[index[:num_test], :]  # 取出洗牌后前 num_test 作为测试集
        target_test = target[index[:num_test]]
        data_train = data[index[num_test:], :]  # 剩余作为训练集
        target_train = target[index[num_test:]]
        return data_train, data_test, target_train, target_test
        pass


    def train_data(self, data_train, data_test, target_train, target_test,kTup=('rbf', 10)):
        # 在这里实现SVM的训练逻辑
        b, alphas = smoP(data_train, target_train, 200, 0.0001, 10, kTup)
        return b,alphas
        pass

    # 测试函数
    def test(self, data_train, data_test, target_train, target_test,b,alphas,kTup=('rbf', 10)):
        datMat = np.mat(data_train);
        labelMat = np.mat(target_train).transpose()
        svInd = np.nonzero(alphas.A > 0)[0]
        sVs = datMat[svInd]
        labelSV = labelMat[svInd];
        #print("支持向量个数:%d" % np.shape(sVs)[0])
        m, n = np.shape(datMat)
        errorCount = 0
        for i in range(m):
            kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
            predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
            if np.sign(predict) != np.sign(target_train[i]): errorCount += 1
        #print("训练集错误率: %.2f%%" % (float(errorCount) / m))



        errorCount = 0
        datMat = np.mat(data_test);
        # labelMat = np.mat(target_test).transpose()
        m, n = np.shape(datMat)
        for i in range(m):
            kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
            predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
            if np.sign(predict) != np.sign(target_test[i]): errorCount += 1
        #print("测试集错误率: %.2f%%" % (float(errorCount) / m))

        print_content="支持向量个数:%d\n训练集错误率: %.2f%%\n测试集错误率: %.2f%%" % (np.shape(sVs)[0],float(errorCount) / m,float(errorCount) / m)
        return print_content
