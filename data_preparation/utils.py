import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import string
import requests
from zipfile import ZipFile

AUTOTUNE = tf.data.AUTOTUNE


class DataPipeline:
    """
    This class reads training data into a pandas.DataFrame then cleans the dataframe with various techniques.
    Finally produces a tensorflow.Data.dataset in zipped format. An instance of this class should be passed to an
    instance of BatchPipeline or can be used as standalone helper.
    """
    dataframe = None  # required for visualization
    dataset = None
    sample_submission_data = None
    submission_test_dataset = None
    input_vectorizer = None
    vocabulary_size = 0
    n_rows = 0
    embedding_dim = None
    # TODO: Decide whether to change these tokens
    tokens = {
        # TODO: Also can extract tweets from the url and append them to the existing one.
        "url": "<URL>",
        "user": "<USER>",
        "smile": "<SMILE>",
        "sadface": "<SAD>",
        "neutralface": "<NEUTRAL>",
        "lolface": "<LOL>",
        "heart": "<LOVE>",
        "number": "<NUMBER>",
        "allcaps": "<ALLCAPS>",
        "hashtag": "<HASHTAG>",
        "repeat": "<REPEAT>",
        "elong": "<ELONG>"
    }

    def __init__(self,
                 train_file_name: str,
                 test_file_name: str,
                 sample_submission_file_name: str,
                 vocabulary_size: int,
                 output_sequence_length=30,
                 glove_embedding_dim=100):
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.sample_submission_file_name = sample_submission_file_name
        self.glove_embedding_dim = glove_embedding_dim
        self.vocabulary_size = vocabulary_size
        self.input_vectorizer = tf.keras.layers.TextVectorization(standardize="lower_and_strip_punctuation",
                                                                  split="whitespace",
                                                                  output_mode="int",
                                                                  max_tokens=vocabulary_size,
                                                                  output_sequence_length=output_sequence_length)

    @staticmethod
    def get_dataframe_from_csv(csv_file_name: str):
        """
        :param csv_file_name: str
        :return: pd.DataFrame()
        """
        if not csv_file_name.endswith(".csv"):
            raise FileNotFoundError('File must be a csv file.')

        file_dir = f'../data/{csv_file_name}'
        print(f"Getting the file: {file_dir}")
        return pd.read_csv(file_dir, sep=',')

    def prepare_dataset(self, include_cols=["location", "keyword"], apply_preprocessing=True):
        """
        loads data from csv and creates a tf.data.Dataset instance containing all data
        :return: tf.data.Dataset
        """
        # get data and create dataset
        dataframe = self.get_dataframe_from_csv(self.train_file_name).fillna(' ')

        # clear the text with pre-determined patterns
        dataframe = self.clear_keywords(dataframe)
        # concatenate location keyword and text then pass to tokenization
        if len(include_cols) > 0:
            for col in include_cols:
                dataframe["text"] = dataframe[col] + " " + dataframe["text"]
        self.dataframe = self.tokenize_dataframe(dataframe) if apply_preprocessing else dataframe
        # we need to remove some unnecessary entries
        print(f"Dataframe size before eliminating too short texts: {len(self.dataframe)}")
        self.dataframe.drop(self.dataframe[self.dataframe["text"].map(lambda entry: len(entry.split(" ")) < 5)].index,
                            inplace=True)
        print(f"Dataframe size after eliminating too short texts: {len(self.dataframe)}")
        print(self.dataframe)
        dataset = self.make_dataset(dataframe["text"], targets=dataframe["target"])
        # get row count
        n_rows = tf.data.experimental.cardinality(dataset).numpy()

        # print some examples of the dataset
        print("-----------------------------------------------------------------------------------------")
        print(f"Dataset \nSize: {n_rows} data points")
        print("Dataset examples:")
        for input_, target in dataset.take(3):
            print(f"Input: {input_}")
            print(f"Target: {target}")
        print("-----------------------------------------------------------------------------------------")

        return dataset

    def prepare_submission_dataset(self, include_cols=["location", "keyword"], apply_preprocessing=True):
        # KAGGLE related stuff
        submission_test_dataframe = self.get_dataframe_from_csv(self.test_file_name).fillna(" ")
        if len(include_cols) > 0:
            for col in include_cols:
                submission_test_dataframe["text"] = submission_test_dataframe[col] + " " + submission_test_dataframe[
                    "text"]
        submission_test_dataframe = self.tokenize_dataframe(
            submission_test_dataframe) if apply_preprocessing else submission_test_dataframe

        # self.vocabulary_size = self.input_vectorizer.vocabulary_size()
        self.sample_submission_data = self.get_dataframe_from_csv(self.sample_submission_file_name)

        return self.make_dataset(submission_test_dataframe["text"])

    def make_dataset(self, inputs: pd.DataFrame, class_num=2, **kwargs):
        """
        :param inputs: pd.DataFrame containing all data.
        :param class_num: int, number of classes to be predicted
        :param kwargs: dict(), used to pass targets: pd.DataFrame
        :return: zipped tf.data.Dataset
        """
        # "tensorize" and vectorize
        inputs = tf.data.Dataset.from_tensor_slices(inputs)
        if self.input_vectorizer is not None:
            self.input_vectorizer.adapt(inputs)
            inputs = inputs.map(self.input_vectorizer)
            print(f"Vocabulary size of the vectorizer: {self.input_vectorizer.vocabulary_size()}")
        # adapter to test.csv file which does not have target column
        if kwargs.get('targets', None) is not None:
            target_vectorizer = tf.keras.layers.CategoryEncoding(num_tokens=class_num, output_mode="one_hot")
            targets = kwargs.get('targets', None).values
            targets = tf.data.Dataset.from_tensor_slices(targets).map(target_vectorizer)
            return tf.data.Dataset.zip((inputs, targets))
        else:
            return tf.data.Dataset.zip((inputs,))

    @staticmethod
    def clear_keywords(dataframe: pd.DataFrame):
        """
        :param dataframe: pandas.DataFrame
        :return: pandas.DataFrame
        """
        for col_name in ["keyword", "location"]:
            clean_keywords = []
            for text in dataframe[col_name].values:
                # non-ascii chars
                if text != '':
                    non_ascii_chars = re.findall(r'[^\x00-\x7F]', text)
                    if len(non_ascii_chars) > 0:
                        text = ''
                    else:
                        # transform encoded space char ('%20') back to a space (' ')
                        text = re.sub(r'%20', " ", re.sub(r' ', "", text))
                        # remove punctuation
                        translator = str.maketrans('', '', string.punctuation)
                        text = text.lower().translate(translator)
                        # clear newline
                        text = re.sub(r'\r', "", re.sub("\n", "", text))
                        # location and keyword should not be too many words
                        words = text.split(' ')
                        if len(words) > 5:
                            text = " ".join(words[0:5])
                    clean_keywords.append(text)
            dataframe[col_name] = clean_keywords
        return dataframe

    def tokenize_dataframe(self, dataframe: pd.DataFrame):
        """
        wrapper function for tokenize
        :param dataframe: a pd.DataFrame whose column `text_col_name` will be tokenized
        :return: pd.DataFrame
        """
        clean_text_col = []
        for text in dataframe["text"].values:
            text = self.tokenize(text)
            clean_text_col.append(text)
        dataframe["text"] = clean_text_col

        return dataframe

    def tokenize(self, text: str, remove_numbers=True, handle_extras=False):
        """
        adapted from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
        in this method we collect what we can and then clean up before passing to the TextVectorization
        layer with standardization
        :param text: text to be tokenized
        :param remove_numbers: bool, removes numbers from the text if true
        :param handle_extras: bool, handles extra stuff, not necessary for this dataset
        :return: str, cleaned tokenized text
        """
        # print(f"Text before tokenization: {text}")

        # clear non-ascii chars
        text = re.sub(r'[^\x00-\x7F]', " ", text)
        # clear newline
        text = re.sub(r'\r', " ", re.sub("\n", "", text))

        # replace mention username with <USER> tag
        text = re.sub(r'@\w+', f" {self.tokens['user']} ", text)
        # handle urls
        text = re.sub(r'https?:\/\/(.+?)(\/.*)', f" {self.tokens['url']} ", text)
        # force splitting words appended with slashes
        text = re.sub(r'/', " / ", text)
        # handle emoticons
        eyes = r"[8:=;]"
        nose = r"['`\-]?"
        # smiley face
        text = re.sub(fr'{eyes}{nose}[)d]+|[)d]+{nose}{eyes}', f" {self.tokens['smile']} ", text)
        # lol face
        text = re.sub(fr'{eyes}{nose}p+', f" {self.tokens['lolface']} ", text)
        # sad face
        text = re.sub(fr'{eyes}{nose}\(+|\)+{nose}{eyes}', f" {self.tokens['sadface']} ", text)
        # neutral face
        text = re.sub(fr'{eyes}{nose}[\/|l*]', f" {self.tokens['neutralface']} ", text)
        # heart
        text = re.sub(r'<3', f" {self.tokens['heart']} ", text)
        # numbers: put space before and after
        # TODO: Maybe parse out time, date etc. and repurpose them?
        # numbers = re.findall(r'[-+]?[.\d]*[\d]+[:,.\d]*', text) -> this regex finds all formats
        numbers = re.findall(r'\d+', text)
        for number in numbers:
            if remove_numbers:
                text = re.sub(rf'({number})', f" ", text)
            else:
                text = re.sub(rf'({number})', f" {number} ", text)
        # hashtag: the regex in the website is buggy, i.e, includes brackets to the hashtag body, so we modify it
        # to capture only alphanumeric characters
        hashtag_regex = r'#[A-Za-z0-9]+'
        # first locate the hashtag
        hashtags = re.findall(hashtag_regex, text)
        # print(f"hashtags: {hashtags}")
        if hashtags is not None:
            # iterate for each match
            for hashtag in hashtags:
                hashtag_replacement = None
                # remove the hashtag char (#)
                hashtag_body = hashtag[1:]
                # skip this check : we do not want to use "hashtag" word unnecessarily
                # first check if the body already contains any tokens
                # for token in self.tokens:
                #     if token in hashtag_body:
                #         hashtag_replacement = token
                #         break

                # if hashtag_replacement is None:
                # check if the hashtag is in all uppercase
                if hashtag_body.isupper():
                    # hashtag_replacement = f" {self.tokens['hashtag']} " + hashtag_body + f" {self.tokens['allcaps']} "
                    hashtag_replacement = hashtag_body
                else:
                    # first split camelCase or PascalCase hashtags into words
                    words = re.split(r'(?=[A-Z])', hashtag_body)
                    # here we remove '' element if existent
                    if '' in words:
                        words.remove('')
                    # hashtag_replacement = f" {self.tokens['hashtag']} " + " ".join(words)
                    hashtag_replacement = " ".join(words)

                text = re.sub(hashtag, hashtag_replacement, text)

        # ############################################## NOT RELEVANT ############################################## #
        # not necessary for this project.
        if handle_extras:
            # punctuation repetitions
            punctuation_repetition_regex = r'([!?.,]){2,}'
            repeated_punctuations = re.findall(punctuation_repetition_regex, text)
            # print(f"Repeated punctuations : {repeated_punctuations}")
            if repeated_punctuations is not None:
                # iterate for each match
                for repeated_punctuation in repeated_punctuations:
                    # replace repeating punctuations with <REPEAT> token
                    # repeated_punctuation_replacement = repeated_punctuation[0] + f" {self.tokens['repeat']} "
                    repeated_punctuation_replacement = repeated_punctuation[0]
                    text = re.sub(punctuation_repetition_regex, repeated_punctuation_replacement, text)
            # elongated words (e.g. heyyyyyyy => hey <ELONG> )
            elongated_words_regex = r'\b(\S*?)(.)\2{2,}\b'
            elongated_words = re.findall(elongated_words_regex, text)
            # print(f"Elongated words: {elongated_words}")
            if elongated_words is not None:
                # iterate for each match
                for elongated_word in elongated_words:
                    # replace elongated word with the <ELONG> token
                    # elongated_word_replacement = elongated_word[0] + elongated_word[1] + f" {self.tokens['elong']} "
                    elongated_word_replacement = elongated_word[0] + elongated_word[1]
                    text = re.sub(elongated_words_regex, elongated_word_replacement, text)

        # ############################################## NOT RELEVANT ############################################## #
        # print(f"before case transform: {text}")

        # remove extra spaces
        text = re.sub(r'[ ]{2,}', ' ', text).strip()

        # remove punctuations, convert to lowercase
        translator = str.maketrans('', '', string.punctuation)
        text = text.lower().translate(translator)

        # print(f"Text after tokenization: {text}")

        return text

    def _load_glove_embeddings(self):
        """
        https://keras.io/examples/nlp/pretrained_word_embeddings/
        :return: dict()
        """
        possible_embedding_dims = [25, 50, 100, 200]
        if self.glove_embedding_dim not in possible_embedding_dims:
            raise ValueError(f"embedding_dim can only be one of these: {possible_embedding_dims}")

        # we go back to the parent which is the root. (This method is called from notebooks/<a notebook>.ipynb)
        notebooks_dir = '/notebooks'
        files_dir = re.sub(notebooks_dir, "", os.getcwd()) + '/glove_embeddings/twitter'
        file_name = f'glove.twitter.27B.{self.glove_embedding_dim}d.txt'

        # download the zip file, then extract if file does not exist
        if not os.path.exists(os.path.join(files_dir, file_name)):
            print("It seems like you do not have the required embeddings for this setting.\n"
                  "The required step are being executed... \n")
            url = 'https://nlp.stanford.edu/data/glove.twitter.27B.zip'
            # first make the directory if it does not exist
            if not os.path.exists(files_dir):
                # mode = 0o666
                os.makedirs(files_dir)
                print(f"Directory created: {files_dir} \n")
            # download the file
            print("Download has started...\n")
            # with a bit of help from https://pythonguides.com/download-zip-file-from-url-using-python/
            # unfortunately, for some reason unzipping without writing the zip to the disk is a bit buggy.
            # therefore, we write the zip to the disk, unzip it, then delete the zip file.
            response = requests.get(url)
            zip_file_name = url.split('/')[-1]
            zip_files_dir = os.path.join(files_dir, zip_file_name)

            with open(zip_files_dir, 'wb') as output_file:
                output_file.write(response.content)
            print("Download completed, now unzipping...\n")
            zip_file = ZipFile(zip_files_dir)
            zip_file.extractall(files_dir)
            print(f"Unzipping completed, extracted all files to {files_dir}. \n")
            # remove the zip file
            os.remove(zip_files_dir)
            print("Removed the zip file")
            print(f"Loading the file: {file_name} \n")

        embeddings_index_map = {}
        with open(os.path.join(files_dir, file_name)) as file:
            for line in file:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index_map[word] = coefs
        print(f"Found {len(embeddings_index_map)} word vectors \n")
        return embeddings_index_map

    def _get_word_index_map(self):
        """
        https://keras.io/examples/nlp/pretrained_word_embeddings/
        :return: dict()
        """
        vocabulary = self.input_vectorizer.get_vocabulary()
        return dict(zip(vocabulary, range(len(vocabulary))))

    def build_embeddings_initializer(self):
        """
        https://keras.io/examples/nlp/pretrained_word_embeddings/
        :return: tf.keras.initializers.Constant()
        """
        glove_embeddings_index_map = self._load_glove_embeddings()
        word_index_map = self._get_word_index_map()
        num_tokens = self.vocabulary_size
        hits = 0
        misses = 0

        # prepare embedding matrix
        embedding_matrix = np.zeros((num_tokens, self.glove_embedding_dim))
        for word, i in word_index_map.items():
            embedding_vector = glove_embeddings_index_map.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        print(f"Converted {hits} words, and missed {misses} words.")

        return tf.keras.initializers.Constant(embedding_matrix)


class BatchPipeline:
    """
    Creates a pipeline for batching the dataset. An instance of this class should be passed to a BaseModel instance.
    """
    dataset = None
    train_dataset = None
    validation_dataset = None
    test_dataset = None
    train_validation_split = 0
    batch_size = 0

    def __init__(self, dataset: tf.data.Dataset, batch_size: int, train_validation_split=0.6):
        if train_validation_split <= 0 or train_validation_split >= 1:
            raise ValueError("The train_validation_split should be between 0 and 1: (0, 1)")
        # shuffle
        n_rows = tf.data.experimental.cardinality(dataset).numpy()
        dataset = dataset.shuffle(n_rows, seed=42, reshuffle_each_iteration=False)
        self.batch_size = batch_size
        self.train_validation_split = train_validation_split
        self.split_data(dataset, n_rows)

    def split_data(self, dataset: tf.data.Dataset, n_rows: int):
        validation_size = round(((1 - self.train_validation_split) / 2) * n_rows)
        self.validation_dataset = dataset.take(validation_size).batch(batch_size=self.batch_size,
                                                                      drop_remainder=True)
        self.validation_dataset = self.validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

        self.test_dataset = dataset.skip(validation_size).take(validation_size).batch(batch_size=self.batch_size,
                                                                                      drop_remainder=True)
        self.test_dataset = self.test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

        self.train_dataset = dataset.skip(2 * validation_size).batch(batch_size=self.batch_size,
                                                                     drop_remainder=True)
        self.train_dataset = self.train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
