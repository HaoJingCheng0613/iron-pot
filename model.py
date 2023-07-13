class Model:
    def __init__(self):
        pass

    def load_data(self, filepath):
        # 这是一个抽象方法，应该在每个子类中实现
        raise NotImplementedError

    def split_data_K_Fold(self, data, test_size):
        # 这是一个抽象方法，应该在每个子类中实现
        raise NotImplementedError

    def split_data_Random(self, data, test_size):
        # 这是一个抽象方法，应该在每个子类中实现
        raise NotImplementedError

    def train_data(self, X_train, y_train):
        # 这是一个抽象方法，应该在每个子类中实现
        raise NotImplementedError




