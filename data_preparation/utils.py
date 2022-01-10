import pandas as pd
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from transformers import BertTokenizer
import tokenization

class DataHandler:
    sample_submission_data = None
    train_dataset = None
    val_dataset = None
    test_dataset = None
    text_tokenizer = None

    def __init__(self,
                 vocabulary_file: np.array,
                 lowercase_file: np.array,
                 train_file_name: str,
                 test_file_name: str,
                 sample_submission_file_name: str,
                 train_split=0.8):
        if train_split <= 0 or train_split > 1:
            raise ValueError('Value of train_split should be in (0,1] inclusive range')
        self.text_tokenizer = TextTokenizer(vocabulary_file, lowercase_file)
        self.prepare_data(train_file_name=train_file_name,
                          test_file_name=test_file_name,
                          sample_submission_file_name=sample_submission_file_name,
                          train_split=train_split)


    @staticmethod
    def get_datatable_from_csv(csv_file_name: str):
        """
        :param csv_file_name: str
        :return: pd.DataFrame()
        """
        file_dir = os.getcwd() + '/data/' + csv_file_name
        print(f"Getting the file: {file_dir}")
        return pd.read_csv(file_dir, sep=',')

    @staticmethod
    def vectorize_labels(train_data: pd.DataFrame, target_col_name: str):
        """
        :param train_data: pd.DataFrame
        :param target_col_name: str
        :return: np.array()
        """
        label = preprocessing.LabelEncoder()
        categorical_data = to_categorical(label.fit_transform(train_data[target_col_name]))
        return categorical_data

    def prepare_data(self, train_file_name: str, test_file_name: str,
                     sample_submission_file_name: str, train_split=0.8, target_col_name='target'):
        if not(train_file_name.endswith('.csv') and test_file_name.endswith('csv')):
            raise FileNotFoundError('File must be a csv file.')
        # get train and test data
        data = self.get_datatable_from_csv(train_file_name).fillna(' ')
        # labels = self.vectorize_labels(data, target_col_name)
        labels = data['target'].values
        test_data = self.get_datatable_from_csv(test_file_name).fillna(' ')
        self.sample_submission_data = self.get_datatable_from_csv(sample_submission_file_name)
        self.train_dataset = {
            'inputs': self.text_tokenizer.tokenize(data),
            'labels': labels
        }
        print(f"Training dataset inputs: {self.train_dataset['inputs'][:6]}")
        print(f"Training dataset labels: {self.train_dataset['labels'][:6]}")
        # self.split_data(data, labels, train_split)
        self.test_dataset = self.text_tokenizer.tokenize(test_data)

    def split_data(self, data: pd.DataFrame, labels: np.array, train_split: float):
        data_row_size = data.shape[0]
        train_size = int(data_row_size * train_split)
        self.train_dataset = {'inputs': data[:train_size], 'labels': labels[:train_size]}
        print(f"Training dataset inputs: {self.train_dataset['inputs'][:6]}")
        print(f"Training dataset labels: {self.train_dataset['labels'][:6]}")
        print(self.train_dataset)
        if train_split < 1:
            self.val_dataset = {'inputs': data[train_size:], 'labels': labels[train_size:]}
            print(f"val_dataset inputs: {self.val_dataset['inputs'][:6]}")
            print(f"val_dataset labels: {self.val_dataset['labels'][:6]}")


class TextTokenizer:
    classification_token = "[CLS]"
    separator_token = "[SEP]"
    tokenizer = None
    bert_tokenizer_name = None
    vocabulary_file = None

    def __init__(self, vocabulary_file: np.array, lowercase_file: np.array,
                 use_model_tokenizer=True, bert_tokenizer_name="bert-base-uncased"):
        # self.vocabulary_file = vocabulary_file
        self.bert_tokenizer_name = bert_tokenizer_name
        print(f"Bert tokenizer: {bert_tokenizer_name}")
        if use_model_tokenizer:
            self.tokenizer = tokenization.FullTokenizer(vocabulary_file, lowercase_file)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_tokenizer_name)

    def tokenize(self, texts: pd.DataFrame, max_length=60):
        all_tokens = []
        all_masks = []
        all_segments = []
        # rearrange the incoming data
        texts = texts['location'] + ' ' + texts['keyword'] + ' ' + texts['text']
        for text in texts:
            text = self.tokenizer.tokenize(text)
            # trim the text so that we have equal length texts
            text = text[:max_length-2]
            input_sequence = [self.classification_token] + text + [self.separator_token]
            tokens, padding_mask, segment_ids = self.encode(input_sequence, max_length)
            all_tokens.append(tokens)
            all_masks.append(padding_mask)
            all_segments.append(segment_ids)

        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

    def encode(self, input_sequence: np.array, max_length: int):
        padding_length = max_length - len(input_sequence)
        # convert tokens to ids
        tokens = self.tokenizer.convert_tokens_to_ids(input_sequence) + [0] * padding_length
        padding_mask = [1 for _ in range(len(input_sequence))] + [0 for _ in range(padding_length)]
        segment_ids = [0 for _ in range(max_length)]

        return tokens, padding_mask, segment_ids
