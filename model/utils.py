import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight


class HyperparameterManager:
    """
    a class for handling hyperparameter operations
    """
    optimizer = None
    class_weights = None
    batch_size = None
    learning_rate = None
    dropout = None
    optimizer_hparams = None
    class_weights_hparams = None
    batch_size_hparams = None
    learning_rate_hparams = None
    dropout_hparams = None

    def __init__(self, hparams: dict()):
        for key in hparams.keys():
            if key.name == 'optimizer':
                self.optimizer = hparams[key]
                self.optimizer_hparams = key
            elif key.name == 'class_weights':
                self.class_weights = hparams[key]
                self.class_weights_hparams = key
            elif key.name == 'batch_size':
                self.batch_size = hparams[key]
                self.batch_size_hparams = key
            elif key.name == 'learning_rate':
                self.learning_rate = hparams[key]
                self.learning_rate_hparams = key
            elif key.name == 'dropout':
                self.dropout = hparams[key]
                self.dropout_hparams = key
            else:
                raise ValueError(f'The key {key} is not defined in this class.')

    def set_hparams(self, optimizer: str, batch_size: int, learning_rate: float, class_weights: str,
                    dropout: float):
        """
        sets the hyperparameters and returns a dict of hparams
        :param optimizer: str, the new optimizer
        :param batch_size: int, the new batch size
        :param learning_rate: float, the new learning rate
        :param class_weights: str, the new class weight choice
        :param dropout: float, the new dropout rate
        :return: dict(), hparams
        """
        # first set the values
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.dropout = dropout
        print(f"Values are set: Optimizer: {self.optimizer} | Learning Rate: {self.learning_rate} | "
              f"Batch Size: {self.batch_size} | Class Weights: {self.class_weights} | Dropout Rate: {self.dropout}")
        # return a dict to be passed to tf.summary
        return {
            self.optimizer_hparams: self.optimizer,
            self.batch_size_hparams: self.batch_size,
            self.learning_rate_hparams: self.learning_rate,
            self.class_weights_hparams: self.class_weights,
            self.dropout_hparams: self.dropout
        }


def get_balanced_class_weights(dataframe: pd.DataFrame):
    """
    method calculates then returns a dict of balanced class weights
    :param dataframe: pd.DataFrame containing the data
    :return: a dict of balanced class weights
    """
    targets = dataframe["target"]
    balanced_class_weights = dict(enumerate(class_weight.compute_class_weight("balanced",
                                                                              classes=targets.unique().tolist(),
                                                                              y=targets.tolist())))
    return balanced_class_weights


def get_angles(pos, i, model_dim):
    """
    calculate angles for positional encoding
    :param pos: int
    :param i: int
    :param model_dim: int,
    :return: np.array of angles
    """
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(model_dim))
    return pos * angle_rates


def positional_encoding(position: int, model_dim: int):
    """
    method creates positional encoding for the attention model
    :param position: int, max position
    :param model_dim: int, model dim
    :return: tf.tensor, encodings
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(model_dim)[np.newaxis, :],
                            model_dim)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# def start_logging(log_directory: str, hparams_list: list, metrics: tf.metrics):
#     """
#     method starts the logging process
#     :param log_directory: the path where the logs will be saved
#     :param hparams_list: list of the hyperparameters that are being logged
#     :param metrics: the metrics that are to be logged
#     """
#     # remove previous logs
#     if os.path.exists(log_directory):
#         shutil.rmtree(log_directory)
#
#     os.makedirs(log_directory)
#
#     with tf.summary.create_file_writer(log_directory).as_default():
#         hp.hparams_config(
#             hparams=hparams_list,
#             metrics=metrics)
