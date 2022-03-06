import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
from tensorboard.plugins.hparams import api as hp
import os
import shutil


class HyperparameterManager:
    optimizer = None
    class_weights = None
    batch_size = None
    learning_rate = None
    dropout = None

    def __init__(self, hparams: dict()):
        for key in hparams.keys():
            if key.name == 'optimizer':
                self.optimizer = hparams[key]
            elif key.name == 'class_weights':
                self.class_weights = hparams[key]
            elif key.name == 'batch_size':
                self.batch_size = hparams[key]
            elif key.name == 'learning_rate':
                self.learning_rate = hparams[key]
            elif key.name == 'dropout':
                self.dropout = hparams[key]
            else:
                raise ValueError(f'The key {key} is not defined in this class.')


def get_balanced_class_weights(dataframe: pd.DataFrame):
    targets = dataframe["target"]
    balanced_class_weights = dict(enumerate(class_weight.compute_class_weight("balanced",
                                                                              classes=targets.unique().tolist(),
                                                                              y=targets.tolist())))
    return balanced_class_weights


def start_logging(log_directory: str, hyperparameters: dict, metrics: list):
    try:
        shutil.rmtree(log_directory)  # clearing logging directory
    except NotADirectoryError:
        pass

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    with tf.summary.create_file_writer(log_directory).as_default():
        hp.hparams_config(
            hparams=hyperparameters.values(),
            metrics=metrics)


def create_submission_for_kaggle(file_name: str, id_list, predictions):
    with open(f"../kaggle_predictions/{file_name}", "w+") as submission_file:
        predictions_dataframe = pd.DataFrame({"id": id_list, "target": predictions})
        predictions_dataframe.to_csv(submission_file, index=False)
    submission_file.close()


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, model_dim):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(model_dim)[np.newaxis, :],
                          model_dim)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)