import pandas as pd
import tensorflow as tf
from sklearn.utils import class_weight
from tensorboard.plugins.hparams import api as hp
import os
import shutil


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
