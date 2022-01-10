import numpy as np
from tensorflow.keras import optimizers


class HyperParameters:
    optimizer = None
    learning_rate = None
    batch_size = None

    def __init__(self, learning_rate: float, optimizer: str, batch_size: int):
        self.learning_rate = learning_rate
        self.optimizer = self.get_optimizer(optimizer)
        self.batch_size = batch_size

    def print(self):
        print("****************** HYPERPARAMETERS ******************")
        print(f'Optimizer: {self.optimizer}')
        print(f'Learning rate: {self.learning_rate}')
        print(f'Batch size: {self.batch_size}')
        print('******************************************************')

    def get_optimizer(self, optimizer: str):
        if optimizer == 'Adam':
            return optimizers.Adam(learning_rate=self.learning_rate)
        elif optimizer == 'Nadam':
            return optimizers.Nadam(learning_rate=self.learning_rate)
        elif optimizer == 'SGD':
            return optimizers.SGD(learning_rate=self.learning_rate)
        else:
            ValueError(f'Given optimizer name: {optimizer} is not implemented.')

    def tune_learning_rate(self, method='random_search', min_val=1e-9, max_val=1e-2):
        if method == 'random_search':
            self.learning_rate = LearningRateTuner.random_search(min_val=min_val, max_val=max_val)
        elif method == 'line_search':
            self.learning_rate = LearningRateTuner.line_search(min_val=min_val, max_val=max_val)


class LearningRateTuner:
    @staticmethod
    def random_search(min_val, max_val):
        return np.random.random(1) * (max_val - min_val) + min_val

    @staticmethod
    def line_search(min_val, max_val):
        # TODO: should return a size of possible values
        return np.random.uniform(min_val, max_val, 1)