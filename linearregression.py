from model import Model

class LinearRegression(Model):

    
    def load_data(self, filepath):
        # 在这里实现线性回归的数据加载逻辑
        pass

    def split_data(self, data, test_size):
        # 在这里实现线性回归的数据分割逻辑
        pass

    def train_data(self, X_train, y_train):
        # 在这里实现线性回归的训练逻辑
        pass

    def test(self):
        print("LinearRegression test")
        pass
