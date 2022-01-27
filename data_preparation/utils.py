import re

import pandas as pd
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
    quiet = None

    def __init__(self,
                 train_file_name: str,
                 test_file_name: str,
                 sample_submission_file_name: str,
                 vocabulary_size=5001,
                 output_sequence_length=120,
                 vectorize=True,
                 quiet=False):
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.sample_submission_file_name = sample_submission_file_name
        self.quiet = quiet
        self.input_vectorizer = layers.TextVectorization(
            standardize="lower_and_strip_punctuation",
            split="whitespace",
            output_mode="int",
            max_tokens=vocabulary_size-1,
            output_sequence_length=output_sequence_length) if vectorize else None

    def get_dataframe_from_csv(self, csv_file_name: str):
        """
        :param csv_file_name: str
        :return: pd.DataFrame()
        """
        if not csv_file_name.endswith(".csv"):
            raise FileNotFoundError('File must be a csv file.')

        file_dir = '../data/' + csv_file_name
        print(f"Getting the file: {file_dir}") if not self.quiet else print()
        return pd.read_csv(file_dir, sep=',')

    def prepare_datasets(self):
        """
        loads data from csv and creates a tf.data.Dataset instance containing all data
        :return: tf.data.Dataset
        """
        # get data and create dataset
        dataframe = self.get_dataframe_from_csv(self.train_file_name).fillna(' ')

        # clear the text with pre-determined patterns
        self.dataframe = self.clear_text(dataframe)
        self.dataframe = self.clear_keywords(self.dataframe)

        dataset = self.make_dataset(dataframe, "target")
        # get row count
        n_rows = tf.data.experimental.cardinality(dataset).numpy()

        # shuffle
        dataset = dataset.shuffle(n_rows, seed=42)

        if not self.quiet:
            # print some examples of the dataset
            print("-----------------------------------------------------------------------------------------")
            print(f"Dataset \nSize: {n_rows}")
            print("Dataset examples:")
            for input_, target in dataset.take(3):
                print(f"Input: {input_}")
                print(f"Target: {target}")
            print("-----------------------------------------------------------------------------------------")

        # KAGGLE related stuff
        submission_test_dataframe = self.get_dataframe_from_csv(self.test_file_name).fillna(" ")
        submission_test_dataframe = self.clear_text(submission_test_dataframe)
        submission_test_dataset = self.make_dataset(submission_test_dataframe, "")

        self.vocabulary_size \
            = self.input_vectorizer.vocabulary_size() + 1 if self.input_vectorizer is not None else None
        self.sample_submission_data = self.get_dataframe_from_csv(self.sample_submission_file_name)

        return dataset, submission_test_dataset

    @staticmethod
    def clear_keywords(dataframe):
        clean_keywords = []
        for keyword in dataframe["keyword"].values:
            keyword = re.sub(r'%20', " ", re.sub(r' ', "", keyword))
            clean_keywords.append(keyword)
        dataframe["keyword"] = clean_keywords

        return dataframe

    def clear_text(self, dataframe):
        clean_text_col = []
        for text in dataframe["text"].values:
            text = self.clear_urls(text)
            text = self.clear_non_ascii(text)
            text = self.clear_newline(text)
            clean_text_col.append(text)
        dataframe["text"] = clean_text_col

        return dataframe

    @staticmethod
    def clear_urls(text):
        # return re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', " ", text)
        return re.sub(r'https?:\/\/(.+?)(\/.*)', " ", text)

    @staticmethod
    def clear_non_ascii(text):
        return re.sub(r'[^\x00-\x7F]', " ", text)

    @staticmethod
    def clear_newline(text):
        return re.sub(r'\r', "", re.sub("\n", "", text))

    def make_dataset(self, dataframe, target_col_name, class_num=2):
        # we do not need "target" column in inputs
        inputs = dataframe.drop(columns=[target_col_name], inplace=False) if target_col_name != "" \
            else dataframe
        # here we build our input by concatenating text data
        inputs = self.build_input_data(inputs)

        # "tensorize" and vectorize
        inputs = tf.data.Dataset.from_tensor_slices(inputs)
        if self.input_vectorizer is not None:
            self.input_vectorizer.adapt(inputs)
            inputs = inputs.map(self.input_vectorizer)
        # adapter to test.csv file which does not have target column
        if target_col_name != '':
            target_vectorizer = layers.CategoryEncoding(num_tokens=class_num, output_mode="one_hot")
            targets = dataframe[target_col_name].values
            targets = tf.data.Dataset.from_tensor_slices(targets).map(target_vectorizer)
            return tf.data.Dataset.zip((inputs, targets))
        else:
            return tf.data.Dataset.zip((inputs,))

    def build_input_data(self, inputs: pd.DataFrame):
        return inputs["location"] + " " + inputs["keyword"] + " " + inputs["text"]


class BatchPipeline:
    dataset = None
    train_dataset = None
    validation_dataset = None
    test_dataset = None
    submission_test_dataset = None
    train_validation_split = 0
    batch_size = 0

    def __init__(self, dataset: tf.data.Dataset, submission_test_dataset: tf.data.Dataset,
                 batch_size: int, train_validation_split=0.8):
        if train_validation_split <= 0 or train_validation_split >= 1:
            raise ValueError("The train_validation_split should be between 0 and 1: (0, 1)")
        self.dataset = dataset
        self.submission_test_dataset = submission_test_dataset
        self.batch_size = batch_size
        self.train_validation_split = train_validation_split
        self.split_data()

    def split_data(self):
        n_rows = tf.data.experimental.cardinality(self.dataset).numpy()
        validation_size = round(((1 - self.train_validation_split) / 2) * n_rows)
        self.validation_dataset = self.dataset.take(validation_size).batch(batch_size=self.batch_size,
                                                                           drop_remainder=True)
        self.validation_dataset = self.validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

        self.test_dataset = self.dataset.skip(validation_size).take(validation_size).batch(batch_size=self.batch_size,
                                                                                           drop_remainder=True)
        self.test_dataset = self.test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

        self.train_dataset = self.dataset.skip(2 * validation_size).batch(batch_size=self.batch_size,
                                                                          drop_remainder=True)
        self.train_dataset = self.train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        if self.submission_test_dataset is not None:
            self.submission_test_dataset = self.submission_test_dataset.batch(batch_size=self.batch_size)
            self.submission_test_dataset = self.submission_test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
