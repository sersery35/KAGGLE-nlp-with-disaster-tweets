import pandas as pd
import os
from tensorflow.keras import layers
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


class DataPipeline:
    dataframe = None  # required for visualization
    dataset = None
    sample_submission_data = None
    submission_test_dataset = None
    input_vectorizer = None
    vocabulary_size = 0
    n_rows = 0

    def __init__(self,
                 train_file_name: str,
                 test_file_name: str,
                 sample_submission_file_name: str,
                 target_col_name="target"):
        self.input_vectorizer = layers.TextVectorization(
            standardize="lower_and_strip_punctuation",
            split="whitespace",
            output_mode="int",
            max_tokens=5000,  # further hyperparams to investigate
            output_sequence_length=140)  # further hyperparams to investigate
        self._prepare_data(train_file_name=train_file_name,
                           test_file_name=test_file_name,
                           sample_submission_file_name=sample_submission_file_name,
                           target_col_name=target_col_name)

    @staticmethod
    def get_dataframe_from_csv(csv_file_name: str):
        """
        :param csv_file_name: str
        :return: pd.DataFrame()
        """
        file_dir = os.getcwd() + '/data/' + csv_file_name
        print(f"Getting the file: {file_dir}")
        return pd.read_csv(file_dir, sep=',')

    @staticmethod
    def target_vectorizer(num_tokens: int, output_mode="one_hot"):
        return layers.CategoryEncoding(num_tokens=num_tokens, output_mode=output_mode)

    def _prepare_data(self, train_file_name: str, test_file_name: str,
                      sample_submission_file_name: str, target_col_name: str):
        """
        loads data from csv and creates a tf.data.Dataset instance containing all data
        :param train_file_name: str
        :param test_file_name: str
        :param sample_submission_file_name: str
        :param target_col_name: str
        :return: None
        """
        if not(train_file_name.endswith('.csv') and test_file_name.endswith('.csv')
               and sample_submission_file_name.endswith(".csv")):
            raise FileNotFoundError('File must be a csv file.')
        # get data and create dataset
        self.dataframe = self.get_dataframe_from_csv(train_file_name).fillna(' ')
        self.dataset = self._make_dataset(self.dataframe, target_col_name)

        # get row count
        self.n_rows = len(self.dataset)

        # shuffle
        self.dataset = self.dataset.shuffle(self.n_rows, seed=42)

        # print some examples of the dataset
        print("-----------------------------------------------------------------------------------------")
        print(f"Dataset \nSize: {self.n_rows}")
        print("Dataset examples:")
        for input_, target in self.dataset.take(3):
            print(f"Input: {input_}")
            print(f"Target: {target}")
        print("-----------------------------------------------------------------------------------------")

        # KAGGLE related stuff
        submission_test_data = self.get_dataframe_from_csv(test_file_name).fillna(' ')
        self.submission_test_dataset = self._make_dataset(submission_test_data, target_col_name='')

        self.vocabulary_size = self.input_vectorizer.vocabulary_size() + 1
        self.sample_submission_data = self.get_dataframe_from_csv(sample_submission_file_name)

    def _make_dataset(self, dataframe: pd.DataFrame, target_col_name: str, class_num=2):
        # we do not need "target" column in inputs
        inputs = dataframe.drop(columns=[target_col_name], inplace=False) if target_col_name != '' else dataframe
        # here we build our input by concatenating text data
        inputs = inputs["location"] + ' ' + inputs["keyword"] + ' ' + inputs["text"]
        # "tensorize" and vectorize
        inputs = tf.data.Dataset.from_tensor_slices(inputs)
        self.input_vectorizer.adapt(inputs)
        inputs = inputs.map(self.input_vectorizer)
        # adapter to test.csv file which does not have target column
        if target_col_name != '':
            targets = dataframe[target_col_name].values
            targets = tf.data.Dataset.from_tensor_slices(targets).map(self.target_vectorizer(class_num))
            return tf.data.Dataset.zip((inputs, targets))
        else:
            return tf.data.Dataset.zip(inputs)


class BatchPipeline:
    dataset = None
    train_dataset = None
    validation_dataset = None
    test_dataset = None
    train_validation_split = 0
    batch_size = 0

    def __init__(self, dataset: tf.data.Dataset, batch_size: int, train_validation_split=0.8):
        if train_validation_split <= 0 or train_validation_split >= 1:
            raise ValueError("The train_validation_split should be between 0 and 1: (0, 1)")
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_validation_split = train_validation_split
        self.n_rows = len(dataset)
        self.split_data()

    def split_data(self):
        validation_size = round(((1 - self.train_validation_split) / 2) * self.n_rows)
        self.validation_dataset = self.dataset.take(validation_size).batch(batch_size=self.batch_size,
                                                                           drop_remainder=True)
        self.validation_dataset = self.validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

        self.test_dataset = self.dataset.skip(validation_size).take(validation_size).batch(batch_size=self.batch_size,
                                                                                           drop_remainder=True)
        self.test_dataset = self.test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

        self.train_dataset = self.dataset.skip(2 * validation_size).batch(batch_size=self.batch_size,
                                                                          drop_remainder=True)
        self.train_dataset = self.train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
