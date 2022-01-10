import keras.losses
from transformers import BertTokenizer
# import tokenization
import tensorflow as tf
import tensorflow_hub as tf_hub
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

from data_preparation.utils import DataHandler
from model.utils import HyperParameters


class FineTunedBertModel:
    bert_module_url = None
    bert_layer = None
    data_handler = None
    model = None
    hyperparameters = None
    model_checkpoint = None

    def __init__(self,
                 train_file_name='train.csv',
                 test_file_name='test.csv',
                 sample_submission_file_name='sample_submission.csv',
                 bert_module_url="https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1",
                 hyperparameters=HyperParameters(optimizer='Adam', learning_rate=1e-5, batch_size=16),
                 model_checkpoint='model.h5'):
        self.data_handler = DataHandler(
            # vocabulary_file=self.bert_layer.resolved_object.vocab_file.asset_path.numpy(),
            train_file_name=train_file_name,
            test_file_name=test_file_name,
            sample_submission_file_name=sample_submission_file_name,
            train_split=1)
        self.bert_module_url = bert_module_url
        self.init_bert_layer()
        self.hyperparameters = hyperparameters
        self.model_checkpoint = ModelCheckpoint(model_checkpoint, monitor='val_loss', save_best_only=True)

    def init_bert_layer(self):
        self.bert_layer = tf_hub.KerasLayer(self.bert_module_url, trainable=True)

    def build(self, max_length=60):
        input_word_ids = layers.Input(shape=(max_length, ), dtype=tf.int32, name='input_word_ids')
        input_mask = layers.Input(shape=(max_length, ), dtype=tf.int32, name='input_mask')
        segment_ids = layers.Input(shape=(max_length, ), dtype=tf.int32, name='segment_ids')
        inputs = [input_word_ids, input_mask, segment_ids]
        _, sequence_output = self.bert_layer(inputs)
        classification_output = sequence_output[:, 0, :]
        outputs = layers.Dense(1, activation='sigmoid')(classification_output)
        self.model = models.Model(inputs=inputs, outputs=outputs)
        self.model.compile(self.hyperparameters.optimizer,
                           loss=keras.losses.binary_crossentropy,
                           metrics=['accuracy'])
        self.hyperparameters.print()
        print(self.model.summary())

    def fit(self, epochs=3, validation_split=0.2):
        self.model.fit(
            x=self.data_handler.train_dataset['inputs'],
            y=self.data_handler.train_dataset['labels'],
            validation_split=validation_split,
            epochs=epochs,
            callbacks=[self.model_checkpoint],
            batch_size=self.hyperparameters.batch_size
        )

    def predict(self):
        self.model.load_weights('model.h5')
        test_predictions = self.model.predict(self.data_handler.test_dataset)
        print(f"Test Predictions: \n{test_predictions}")
        return test_predictions

