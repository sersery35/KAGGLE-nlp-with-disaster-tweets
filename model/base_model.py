import numpy as np
from tensorflow import keras, nn
from official.nlp import optimization
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf


class BaseModel:
    batch_pipeline = None
    train_dataset = None
    validation_dataset = None
    test_dataset = None
    hyperparameters = None
    hparams = None
    optimizer = None
    model = None
    run_name = None
    class_weights = None
    session_num = 0
    history = None

    def __init__(self, batch_pipeline, parameters: dict, hyperparameters: dict, hparams: dict,
                 class_weights: dict):
        self.parameters = parameters
        self.hyperparameters = hyperparameters
        self.hparams = hparams
        self.class_weights \
            = class_weights if self.hparams[self.hyperparameters["class_weights"]] == "balanced" else None
        self.batch_pipeline = batch_pipeline
        self._init_datasets()
        self._set_optimizer()
        self._set_model()

    def _init_datasets(self):
        # unpack for convenience
        self.train_dataset = self.batch_pipeline.train_dataset
        self.validation_dataset = self.batch_pipeline.validation_dataset
        self.test_dataset = self.batch_pipeline.test_dataset

    def _set_optimizer(self):
        train_dataset_len = tf.data.experimental.cardinality(self.batch_pipeline.train_dataset).numpy()
        if self.hparams[self.hyperparameters["optimizer"]] == "sgd":
            self.optimizer = keras.optimizers.SGD(learning_rate=self.hparams[self.hyperparameters["learning_rate"]])
        elif self.hparams[self.hyperparameters["optimizer"]] == "adam":
            self.optimizer = keras.optimizers.Adam(learning_rate=self.hparams[self.hyperparameters["learning_rate"]])
        elif self.hparams[self.hyperparameters["optimizer"]] == "adamw":
            self.optimizer = optimization.create_optimizer(
                init_lr=self.hparams[self.hyperparameters["learning_rate"]],
                num_train_steps=self.parameters["epochs"] * train_dataset_len,
                num_warmup_steps=round(0.1 * self.parameters["epochs"] * train_dataset_len),
                optimizer_type="adamw")
        else:
            raise ValueError(f"No implementation exists for the given optimizer: "
                             f"{self.hparams[self.hyperparameters['optimizer']]}")

    def _set_run_name(self):
        self.run_name = f"run={self.session_num}" \
                        f"__lr={self.hparams[self.hyperparameters['learning_rate']]}" \
                        f"__hidden_unit={self.hparams[self.hyperparameters['hidden_unit']]}" \
                        f"__batch_size={self.batch_pipeline.batch_size}" \
                        f"__optimizer={self.hparams[self.hyperparameters['optimizer']]}" \
                        f"__class_weights={self.hparams[self.hyperparameters['class_weights']]}" \
                        f"__dropout={self.hparams[self.hyperparameters['dropout']]}"

    def _set_model(self):
        self._set_run_name()
        self.model = keras.Sequential([
            keras.layers.Embedding(self.parameters["vocabulary_size"], self.parameters["embedding_dim"],
                                   mask_zero=True),
            keras.layers.Bidirectional(keras.layers.LSTM(16, return_sequences=True)),
            keras.layers.Dense(self.hparams[self.hyperparameters["hidden_unit"]], activation=nn.leaky_relu),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dropout(self.hparams[self.hyperparameters["dropout"]]),
            keras.layers.Dense(self.parameters["n_labels"], activation=nn.sigmoid)
        ])

        self.model.compile(optimizer=self.optimizer,
                           loss=keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        print(self.model.summary())

    def fit_and_evaluate(self, log_directory):
        print(f"{self.run_name} starting...")

        self.history = self.model.fit(
            self.train_dataset,
            validation_data=self.validation_dataset,
            epochs=self.parameters["epochs"],
            callbacks=[keras.callbacks.TensorBoard(log_dir=f"{log_directory}{self.run_name}",
                                                   histogram_freq=1,
                                                   update_freq="batch")],
            class_weight=self.class_weights,
        )

        test_loss, test_accuracy = self.model.evaluate(self.test_dataset)
        true_labels = np.concatenate([y for x, y in self.test_dataset], axis=0).argmax(axis=-1)
        predictions = self.model.predict(self.test_dataset).argmax(axis=-1)
        print(f"{self.run_name} completed.")
        precision, recall, f1, support = precision_recall_fscore_support(true_labels, predictions, average="macro")

        return test_accuracy, precision, recall, f1

    def predict_for_kaggle(self, kaggle_test_dataset: tf.data.Dataset):
        return self.model.predict(kaggle_test_dataset).argmax(axis=-1)