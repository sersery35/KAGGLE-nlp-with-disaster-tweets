import numpy as np
from tensorflow import keras, nn
from official.nlp import optimization
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import os


class BaseModel:
    """
    a wrapper class for handling model operations
    """
    name = None
    hparam_manager = None
    batch_pipeline = None
    epochs = None
    run_name = None
    # outputs
    model = None
    run_name = None
    history = None

    def __init__(self, vocabulary_size: int, embedding_dim: int, lstm_dims: list, hidden_dim: int, num_classes: int,
                 epochs, batch_pipeline, hparam_manager, embeddings_initializer=tf.keras.initializers.GlorotUniform(),
                 embeddings_regularizer=None):
        self.name = 'basemodel'
        self.batch_pipeline = batch_pipeline
        self.epochs = epochs
        self.hparam_manager = hparam_manager
        train_dataset_len = tf.data.experimental.cardinality(batch_pipeline.train_dataset).numpy()
        optimizer = self._get_optimizer(train_dataset_len)
        self._set_model(vocabulary_size, embedding_dim, lstm_dims, hidden_dim, num_classes, optimizer,
                        embeddings_initializer, embeddings_regularizer)

    def _get_optimizer(self, train_dataset_len: int):
        """
        method sets and returns the optimizer
        :param train_dataset_len: the length of the train dataset
        :return: tf.keras.optimizers instance
        """
        if self.hparam_manager.optimizer == "sgd":
            return keras.optimizers.SGD(learning_rate=self.hparam_manager.learning_rate)
        elif self.hparam_manager.optimizer == "adam":
            return keras.optimizers.Adam(learning_rate=self.hparam_manager.learning_rate)
        elif self.hparam_manager.optimizer == "adamw":
            return optimization.create_optimizer(
                init_lr=self.hparam_manager.learning_rate,
                num_train_steps=self.epochs * train_dataset_len,
                num_warmup_steps=round(0.1 * self.epochs * train_dataset_len),
                optimizer_type="adamw")
        else:
            raise ValueError(f"No implementation exists for the given optimizer: "
                             f"{self.hparam_manager.optimizer}")

    def _set_run_name(self):
        self.run_name = f"basemodel" \
                        f"__lr={self.hparam_manager.learning_rate}" \
                        f"__batch_size={self.batch_pipeline.batch_size}" \
                        f"__optimizer={self.hparam_manager.optimizer}" \
                        f"__class_weights={self.hparam_manager.class_weights or 'None'}" \
                        f"__dropout={self.hparam_manager.dropout}"

    def _set_model(self, vocabulary_size: int, embedding_dim: int, lstm_dims: int, hidden_dim: int, num_classes: int,
                   optimizer: tf.keras.optimizers, embeddings_initializer: tf.keras.initializers,
                   embeddings_regularizer: tf.keras.regularizers):
        """
        method sets the model
        :param vocabulary_size: the size of the vocabulary
        :param embedding_dim: embedding layer dim
        :param lstm_dims: list of dimensions of lstm layers; for each value an LSTM layer is created.
        :param hidden_dim: dense layer dim
        :param num_classes: number of classes
        :param optimizer: optimizer of the model
        :param embeddings_initializer: embeddings_initializer of the Embedding layer
        :param embeddings_regularizer: embeddings_regularizer of the Embedding layer, initially None
        """
        is_trainable = mask_zero = not(isinstance(embeddings_initializer, tf.keras.initializers.Constant))
        embedding_layer = keras.layers.Embedding(vocabulary_size, embedding_dim,
                                                 embeddings_initializer=embeddings_initializer,
                                                 embeddings_regularizer=embeddings_regularizer,
                                                 trainable=is_trainable,
                                                 mask_zero=mask_zero)
        all_layers = [embedding_layer]
        # dropout after embedding
        dropout_layer = keras.layers.Dropout(self.hparam_manager.dropout)
        all_layers.append(dropout_layer)
        # bidirectional lstms
        bidirectional_lstm_layers = [keras.layers.Bidirectional(keras.layers.LSTM(lstm_dim, return_sequences=True))
                                     for lstm_dim in lstm_dims]
        all_layers.extend(bidirectional_lstm_layers)
        # first dense after lstms
        dense_layer = keras.layers.Dense(hidden_dim, activation=nn.relu) # could be leaky_relu
        all_layers.append(dense_layer)
        # pooling
        pooling_layer = keras.layers.GlobalAveragePooling1D()
        all_layers.append(pooling_layer)
        # dropout after pooling
        dropout_layer = keras.layers.Dropout(self.hparam_manager.dropout)
        all_layers.append(dropout_layer)
        # classification
        classification_layer = keras.layers.Dense(num_classes, activation=nn.sigmoid)  # could be sigmoid too.
        all_layers.append(classification_layer)

        self.model = keras.Sequential(all_layers)

        self.model.compile(optimizer=optimizer,
                           loss=keras.losses.BinaryCrossentropy(),
                           metrics=['accuracy'])
        self._set_run_name()
        print(self.model.summary())

    def fit_and_evaluate(self, class_weights: dict(), log_directory: str):
        """
        wrapper method for model.fit() and model.evaluate()
        :param class_weights: a dict() that contains the balanced class weights
        :param log_directory: the directory to create the logs for this model
        :return: test_accuracy, precision, recall, f1, predictions
        """

        print(f"{self.run_name} starting...")
        res = self.model.fit(
            self.batch_pipeline.train_dataset,
            validation_data=self.batch_pipeline.validation_dataset,
            epochs=self.epochs,
            callbacks=[keras.callbacks.TensorBoard(log_dir=f"{log_directory}{self.run_name}",
                                                   histogram_freq=1,
                                                   update_freq="batch"),
                       keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)],
            class_weight=class_weights if self.hparam_manager.class_weights == "balanced" else None)
        self.history = res.history
        test_loss, test_accuracy = self.model.evaluate(self.batch_pipeline.test_dataset)
        true_labels = np.concatenate([y for x, y in self.batch_pipeline.test_dataset], axis=0).argmax(axis=-1)
        predictions = self.model.predict(self.batch_pipeline.test_dataset).argmax(axis=-1)
        print(f"{self.run_name} completed.")
        precision, recall, f1, support = precision_recall_fscore_support(true_labels, predictions, average="macro")

        return test_accuracy, precision, recall, f1, predictions

    def test_model(self, dataset: tf.data.Dataset, epochs: int):
        """
        method tries to overfit with a small dataset to make a sanity check
        """
        self.model.fit(
            dataset,
            validation_data=dataset,
            epochs=epochs
        )

    def save_model(self, file_path='../model/models/'):
        """
        method saves the current model in file_path named as self.name
        :param file_path: str, the path to store the model
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, self.name)
        tf.keras.models.save_model(self.model, file_path)

    def predict_for_kaggle(self, kaggle_test_dataset: tf.data.Dataset):
        """
        makes a prediction for the kaggle test dataset with the current model
        :param kaggle_test_dataset: a tensorflow dataset
        :return: np.array() of predictions
        """
        return self.model.predict(kaggle_test_dataset).argmax(axis=-1)
