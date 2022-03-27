import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import string
import requests
from zipfile import ZipFile

AUTOTUNE = tf.data.AUTOTUNE


# https://nlp.stanford.edu/projects/glove/ for glove related

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
    embedding_dim = None
    glove_url = None
    tokens = {
        "url": "<URL>",
        "user": "<USER>",
        "smile": "<SMILE>",
        "sadface": "<SAD>",
        "neutralface": "<NEUTRAL>",
        "lolface": "<LAUGH>",
        "heart": "<LOVE>",
        "number": "<NUMBER>",
        "hashtag": "<HASHTAG>",
    }

    def __init__(self,
                 train_file_name: str,
                 kaggle_test_file_name: str,
                 sample_submission_file_name: str,
                 max_vocabulary_size: int,
                 output_sequence_length=30,
                 glove_embedding_dim=100,
                 glove_url='https://nlp.stanford.edu/data/glove.twitter.27B.zip'):
        # check if the glove_embedding_dim is a possible embedding dim
        assert glove_embedding_dim in [25, 50, 100, 200, 300]
        self.train_file_name = train_file_name
        self.kaggle_test_file_name = kaggle_test_file_name
        self.sample_submission_data = self._get_dataframe_from_csv(sample_submission_file_name)
        self.glove_embedding_dim = glove_embedding_dim
        self.glove_url = glove_url
        self.input_vectorizer = tf.keras.layers.TextVectorization(standardize="lower_and_strip_punctuation",
                                                                  split="whitespace",
                                                                  output_mode="int",
                                                                  max_tokens=max_vocabulary_size,
                                                                  output_sequence_length=output_sequence_length)

    def prepare_train_dataset(self, include_cols=["location", "keyword"], extract_extras=True, class_num=2,
                              vectorize=True, drop_limit=5):
        """
        loads train data from csv and creates a tf.data.Dataset instance containing all training data
        :param include_cols: which columns to include into the text (the data to be procesed)
        :param extract_extras: whether to apply a custom method that extracts feelings and other valuable
        attributes from the text
        :param class_num: number of classes
        :param vectorize: whether to use TextVectorization layer to vectorize the data
        :param drop_limit: the minimum numbers of tokens (words, elements) required to have in order to be included in
        training process
        :return: tf.data.Dataset
        """
        dataset, self.dataframe = self._prepare(self.train_file_name,
                                                include_cols=include_cols,
                                                extract_extras=extract_extras,
                                                class_num=class_num,
                                                vectorize=vectorize,
                                                drop_limit=drop_limit)

        return dataset

    def prepare_submission_dataset(self, include_cols=["location", "keyword"], extract_extras=True, class_num=2,
                                   vectorize=True, drop_limit=5):
        """
        prepares the dataset for submission to KAGGLE
        :param include_cols: which columns to include into the text (the data to be procesed)
        :param extract_extras: whether to apply a custom method that extracts feelings and other valuable
        attributes from the text
        :param class_num: number of classes
        :param vectorize: whether to use TextVectorization layer to vectorize the data
        :param drop_limit: the minimum numbers of tokens (words, elements) required to have in order to be included in
        training process
        :return: a tensorflow dataset
        """
        # KAGGLE related stuff
        return self._prepare(self.kaggle_test_file_name,
                             include_cols=include_cols,
                             extract_extras=extract_extras,
                             class_num=class_num,
                             vectorize=vectorize,
                             drop_limit=drop_limit,
                             include_targets=False)

    def _prepare(self, file_name: str, include_cols=["location", "keyword"], extract_extras=True, class_num=2,
                 vectorize=True, drop_limit=5, include_targets=True):
        """
        method loads the data into a pd.DataFrame then applies several methods as preprocessing then finally returns a
        tf.data.Dataset containing all inputs and targets
        :param file_name: file to be loaded into a dataset
        :param include_cols: which columns to include into the text (the data to be procesed)
        :param extract_extras: whether to apply a custom method that extracts feelings and other valuable
        attributes from the text
        :param class_num: number of classes
        :param vectorize: whether to use TextVectorization layer to vectorize the data
        :param drop_limit: the minimum numbers of tokens (words, elements) required to have in order to be included in
        training process
        :return: tf.data.Dataset, pd.DataFrame
        """
        # get data and create dataset
        dataframe = self._get_dataframe_from_csv(file_name).fillna(' ')
        # clear the text with pre-determined patterns
        dataframe = self._clear_keywords(dataframe)
        # concatenate location keyword and text then pass to tokenization
        if len(include_cols) > 0:
            for col in include_cols:
                dataframe["text"] = dataframe[col] + " " + dataframe["text"]
        dataframe = self._extract_extras_from_dataframe(dataframe) if extract_extras else dataframe
        # we need to remove some unnecessary entries
        print(f"Dataframe size before eliminating too short texts: {len(dataframe)}")
        dataframe.drop(dataframe[dataframe["text"].map(lambda entry: len(entry.split(" ")) < drop_limit)].index,
                       inplace=True)
        print(f"Dataframe size after eliminating too short texts: {len(dataframe)}")
        print(dataframe)
        dataset = self._make_dataset(dataframe["text"],
                                     targets=dataframe["target"] if include_targets else None,
                                     class_num=class_num,
                                     vectorize=vectorize,
                                     predict=not include_targets)
        # get row count
        n_rows = tf.data.experimental.cardinality(dataset).numpy()

        # print some examples of the dataset
        if include_targets:
            print("-----------------------------------------------------------------------------------------")
            print(f"Dataset \nSize: {n_rows} data points")
            print("Dataset examples:")
            for input_, target in dataset.take(3):
                print(f"Input: {input_}")
                print(f"Target: {target}")
            print("-----------------------------------------------------------------------------------------")

        return dataset, dataframe

    def _make_dataset(self, inputs: pd.DataFrame, targets=None, class_num=2, vectorize=True, predict=False):
        """
        creates a tf.data.Dataset ready to be passed to the model as input.
        :param inputs: pd.DataFrame containing all data.
        :param targets: pd.DataFrame containing all label data
        :param class_num: int, number of classes to be predicted
        :param vectorize: bool, indicates whether to use the TextVectorization layer. if this is set to false, then
        a TextVectorizer should be provided to the model
        :return: zipped tf.data.Dataset
        """
        # "tensorize" and vectorize
        inputs = tf.data.Dataset.from_tensor_slices(inputs.values)
        if vectorize:
            if not predict:
                self.input_vectorizer.adapt(inputs)
            inputs = inputs.map(self.input_vectorizer)
            print(f"Vocabulary size of the vectorizer: {self.input_vectorizer.vocabulary_size()}")
            self.vocabulary_size = self.input_vectorizer.vocabulary_size()

        # adapter to test.csv file which does not have target column
        if targets is not None:
            targets = tf.data.Dataset.from_tensor_slices(targets.values)
            target_encoder = tf.keras.layers.CategoryEncoding(num_tokens=class_num, output_mode="one_hot")
            targets = targets.map(target_encoder)
            return tf.data.Dataset.zip((inputs, targets))
        else:
            return tf.data.Dataset.zip((inputs,))

    @staticmethod
    def _get_dataframe_from_csv(csv_file_name: str):
        """
        :param csv_file_name: str
        :return: pd.DataFrame()
        """
        if not csv_file_name.endswith(".csv"):
            raise FileNotFoundError('File must be a csv file.')

        file_dir = f'../data/{csv_file_name}'
        print(f"Getting the file: {file_dir}")
        return pd.read_csv(file_dir, sep=',')

    @staticmethod
    def _clear_keywords(dataframe: pd.DataFrame):
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

    @staticmethod
    def _extract_urls(dataframe, silent=True):
        """
        method extracts urls from a dataframe -> not needed for this project since we can not use tweepy API to its full
        extent, i.e, get tweets by id
        :param dataframe: pd.DataFrame, dataframe to be processed
        :param silent: bool, prints out the extracted urls if True
        :return: pd.DataFrame with an extra url column with the corresponding urls
        """
        # capture url domain and dir, join them to create a downloadable link then remove quotation marks
        urls = dataframe["text"].map(lambda text: ["".join([re.sub(r'[\"\']', "", element) for element in group])
                                                   for group in re.findall(r'(https?:\/\/)(\S+?)(\/\S+)', text)])
        # print out the extracted urls
        if not silent:
            for row in urls.values:
                if len(row) > 0:
                    print(f"{row}")
        dataframe["url"] = urls
        return dataframe

    def _extract_extras_from_dataframe(self, dataframe: pd.DataFrame, remove_numbers=False):
        """
        wrapper function for _extract_extras
        :param dataframe: a pd.DataFrame whose column `text_col_name` will be tokenized
        :param remove_numbers: bool, removes numbers from the text if true
        :return: pd.DataFrame
        """
        clean_text_col = []
        for text in dataframe["text"].values:
            text = self._extract_extras(text, remove_numbers=remove_numbers)
            clean_text_col.append(text)
        dataframe["text"] = clean_text_col

        return dataframe

    def _extract_extras(self, text: str, remove_numbers: bool):
        """
        adapted from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
        in this method we collect what we can and then clean up before passing to the TextVectorization
        layer with standardization
        :param text: text to be tokenized
        :param remove_numbers: bool, removes numbers from the text if true
        :return: str, cleaned more insightful text
        """
        # print(f"Text before tokenization: {text}")

        # clear non-ascii chars
        text = re.sub(r'[^\x00-\x7F]', " ", text)
        # clear newline
        text = re.sub(r'\r', " ", re.sub("\n", "", text))
        # replace mention username with <USER> tag
        text = re.sub(r'@\w+', f" {self.tokens['user']} ", text)
        # handle urls
        text = re.sub(r'(https?:\/\/)(\S+?)(\/\S+)', f" {self.tokens['url']} ", text)
        # force splitting words appended with slashes
        text = re.sub(r'/', " / ", text)
        # handle emoticons
        eyes = r'[8:=;]'
        nose = r'[\'`\-]?'
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
        numbers = re.findall(r'\d+', text)
        for number in numbers:
            if remove_numbers:
                text = re.sub(rf'({number})', " ", text)
            else:
                text = re.sub(rf'({number})', f" {number} ", text)
        # hashtag: the regex in the website is buggy, i.e, includes brackets to the hashtag body, so we modify it
        # to capture only alphanumeric characters
        hashtag_regex = r'#[A-Za-z0-9]+'
        # first locate the hashtag
        hashtags = re.findall(hashtag_regex, text)

        if hashtags is not None:
            # iterate for each match
            for hashtag in hashtags:
                # remove the hashtag char (#)
                hashtag_body = hashtag[1:]
                # check if the hashtag is in all uppercase
                if hashtag_body.isupper():
                    hashtag_replacement = hashtag_body
                else:
                    # first split camelCase or PascalCase hashtags into words
                    words = re.split(r'(?=[A-Z])', hashtag_body)
                    # here we remove '' element if existent
                    if '' in words:
                        words.remove('')
                    hashtag_replacement = " ".join(words)
                text = re.sub(hashtag, hashtag_replacement, text)

        # punctuation repetitions
        punctuation_repetition_regex = r'([!?.,]){2,}'
        repeated_punctuations = re.findall(punctuation_repetition_regex, text)
        # print(f"Repeated punctuations : {repeated_punctuations}")
        if repeated_punctuations is not None:
            # iterate for each match
            for repeated_punctuation in repeated_punctuations:
                # remove repeating punctuations
                repeated_punctuation_replacement = repeated_punctuation[0]
                text = re.sub(punctuation_repetition_regex, repeated_punctuation_replacement, text)
        # elongated words (e.g. heyyyyyyy => hey)
        elongated_words_regex = r'\b(\S*?)(.)\2{2,}\b'
        elongated_words = re.findall(elongated_words_regex, text)
        # print(f"Elongated words: {elongated_words}")
        if elongated_words is not None:
            # iterate for each match
            for elongated_word in elongated_words:
                # replace elongated word with the original form
                elongated_word_replacement = elongated_word[0] + elongated_word[1]
                text = re.sub(elongated_words_regex, elongated_word_replacement, text)
        # remove extra spaces
        text = re.sub(r'[ ]{2,}', ' ', text).strip()
        # remove punctuations, convert to lowercase
        translator = str.maketrans('', '', string.punctuation)
        text = text.lower().translate(translator)

        return text

    def _load_glove_embeddings(self):
        """
        method loads the glove embeddings into a dict
        https://keras.io/examples/nlp/pretrained_word_embeddings/
        :return: dict()
        """
        # we go back to the parent which is the root. (This method is called from notebooks/<a notebook>.ipynb)
        notebooks_dir = '/notebooks'
        # set the file directories and file names by parsing the url
        zip_file_name = self.glove_url.split('/')[-1]
        glove_type = re.match(r'glove\.([A-Za-z0-9\.]+)\.zip', zip_file_name).group(1)
        file_name = f'glove.{glove_type}.{self.glove_embedding_dim}d.txt'
        files_dir = re.sub(notebooks_dir, "", os.getcwd()) + f'/glove_embeddings/{glove_type}'

        # download the zip file, then extract if file does not exist
        if not os.path.exists(os.path.join(files_dir, file_name)):
            self._download_and_load_glove_embedding_file(files_dir, zip_file_name)

        # fetch possible embedding dimensions from the file directory and do a check
        possible_embedding_dims = [int(re.match(r'glove\.([A-Za-z0-9]+[\.])*(\d+)d\.txt', d).group(2))
                                   for d in os.listdir(files_dir)]
        if self.glove_embedding_dim not in possible_embedding_dims:
            raise ValueError(f"embedding_dim can only be one of these: {possible_embedding_dims}")

        print(f"Loading the file: {file_name} \n")

        embeddings_index_map = {}
        with open(os.path.join(files_dir, file_name)) as file:
            for line in file:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index_map[word] = coefs
        print(f"Found {len(embeddings_index_map)} word vectors \n")
        return embeddings_index_map

    def _download_and_load_glove_embedding_file(self, files_dir: str, zip_file_name: str):
        """
        method downloads the file in the glove_url then extracts the components of the zip under files_dir
        then deletes the zip file
        :param files_dir: the file directory to save the files
        :param zip_file_name: name of the zip file
        """
        print("It seems like you do not have the required embeddings for this setting.\n"
              "The required step are being executed... \n")
        # first make the directory if it does not exist
        if not os.path.exists(files_dir):
            os.makedirs(files_dir)
            print(f"Directory created: {files_dir} \n")
        # download the file
        print("Download has started...\n")
        # with a bit of help from https://pythonguides.com/download-zip-file-from-url-using-python/
        # unfortunately, for some reason unzipping without writing the zip to the disk is a bit buggy.
        # therefore, we write the zip to the disk, unzip it, then delete the zip file.
        response = requests.get(self.glove_url)

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

    def _get_word_index_map(self):
        """
        https://keras.io/examples/nlp/pretrained_word_embeddings/
        :return: dict()
        """
        vocabulary = self.input_vectorizer.get_vocabulary()
        return dict(zip(vocabulary, range(len(vocabulary))))

    def _build_embeddings_initializer_for_glove(self):
        """
        method builds the embedding matrix using pre-trained embeddings
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

    def build_embeddings_initializer(self):
        """
        return the required Embedding initializer
        :return:tf.keras.initializers
        """
        if self.glove_embedding_dim is None:
            return tf.keras.initializers.GlorotUniform()
        else:
            return self._build_embeddings_initializer_for_glove()

    @staticmethod
    def build_embeddings_regularizer(strength=0.0001):
        return tf.keras.regularizers.l1(strength)


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
        if train_validation_split <= 0 or train_validation_split > 1:
            raise ValueError("The train_validation_split should be between 0 and 1: (0, 1)")
        # shuffle
        n_rows = len(dataset)
        dataset = dataset.shuffle(n_rows, seed=42, reshuffle_each_iteration=False)
        self.batch_size = batch_size
        self.train_validation_split = train_validation_split
        self.split_data(dataset, n_rows)

    def split_data(self, dataset: tf.data.Dataset, n_rows: int):
        """
        method sets the train_dataset, validation_dataset, and test_dataset of this class.
        these datasets are used when the BatchPipeline is passed to the model.
        :param dataset: a tensorflow dataset to be split into train, val and test
        :param n_rows: precalculated number of rows, passed for convenience
        """
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
