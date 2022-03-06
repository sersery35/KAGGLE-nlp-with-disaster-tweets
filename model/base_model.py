import numpy as np
from tensorflow import keras, nn
from official.nlp import optimization
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf


class BaseModel:
    # inputs
    vocabulary_size = None
    embedding_dim = None
    lstm_dim = None
    hidden_dim = None
    embeddings_initializer = None
    optimizer = None
    class_weights = None
    epochs = None
    n_labels = None
    batch_pipeline = None
    train_dataset = None
    validation_dataset = None
    test_dataset = None
    hparam_manager = None
    # outputs
    model = None
    run_name = None
    history = None

    def __init__(self, vocabulary_size: int, embedding_dim: int, lstm_dim: int, hidden_dim: int, n_labels: int,
                 encoder: tf.keras.layers.TextVectorization, epochs: int, batch_pipeline, hparam_manager,
                 class_weights: dict, embeddings_initializer=tf.keras.initializers.GlorotUniform()):
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.hidden_dim = hidden_dim
        self.embeddings_initializer = embeddings_initializer
        self.n_labels = n_labels
        self.epochs = epochs
        self.hparam_manager = hparam_manager
        self.class_weights = class_weights if self.hparam_manager.class_weights == "balanced" else None
        self.batch_pipeline = batch_pipeline
        optimizer = self._set_optimizer()
        self._set_model(optimizer, encoder)

    def _set_optimizer(self):
        train_dataset_len = tf.data.experimental.cardinality(self.batch_pipeline.train_dataset).numpy()
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
        self.run_name = f"run -> " \
                        f"__lr={self.hparam_manager.learning_rate}" \
                        f"__batch_size={self.batch_pipeline.batch_size}" \
                        f"__optimizer={self.hparam_manager.optimizer}" \
                        f"__class_weights={self.hparam_manager.class_weights or 'None'}" \
                        f"__dropout={self.hparam_manager.dropout}"

    def _set_model(self, optimizer, encoder):

        self._set_run_name()
        self.model = keras.Sequential([
            keras.layers.Embedding(self.vocabulary_size, self.embedding_dim, self.embeddings_initializer, mask_zero=True),
            keras.layers.Dropout(self.hparam_manager.dropout),
            keras.layers.Bidirectional(keras.layers.LSTM(self.lstm_dim, return_sequences=True)),
            keras.layers.Dense(self.hidden_dim, activation=nn.relu),  # could be leaky_relu
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(self.n_labels, activation=nn.softmax)  # could be sigmoid too.
        ])

        self.model.compile(optimizer=optimizer,
                           loss=keras.losses.BinaryCrossentropy(),
                           metrics=['accuracy'])
        print(self.model.summary())

    def fit_and_evaluate(self, log_directory):
        """
        wrapper method for model.fit() and model.evaluate()
        :param log_directory:
        :return:
        """
        print(f"{self.run_name} starting...")

        res = self.model.fit(
            self.batch_pipeline.train_dataset,
            validation_data=self.batch_pipeline.validation_dataset,
            epochs=self.epochs,
            callbacks=[keras.callbacks.TensorBoard(log_dir=f"{log_directory}{self.run_name}",
                                                   histogram_freq=1,
                                                   update_freq="batch")],
            class_weight=self.class_weights)
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

    def predict_for_kaggle(self, kaggle_test_dataset: tf.data.Dataset):
        """
        makes a prediction for the kaggle test dataset with the current model
        :param kaggle_test_dataset: a tensorflow dataset
        :return: np.array() of predictions
        """
        return self.model.predict(kaggle_test_dataset).argmax(axis=-1)
