import keras
import tensorflow as tf
import tensorflow_hub as tf_hub
from .base_model import BaseModel


# TODO: THIS MODEL IS NOT FINE-TUNED YET


class SimpleBertModel(BaseModel):
    """
    this class expects non-vectorized inputs, i.e, set vectorize=False for the DataPipeline instance which is then
    passed to BatchPipeline that is expected in the initialization of this class.
    """
    bert_model_url = None
    bert_preprocessor_url = None
    bert_layer = None

    def __init__(self, batch_pipeline, parameters: dict, hyperparameters: dict, hparams: dict,
                 class_weights: dict,
                 bert_model_url="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1",
                 bert_preprocessor_url="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"):
        self.bert_model_url = bert_model_url
        self.bert_preprocessor_url = bert_preprocessor_url
        self.parameters = parameters
        self.hyperparameters = hyperparameters
        self.hparams = hparams
        self.class_weights \
            = class_weights if self.hparams[self.hyperparameters["class_weights"]] == "balanced" else None
        self.batch_pipeline = batch_pipeline
        self._init_datasets()
        self._set_optimizer()
        self._set_model()

    def _set_model(self):
        self._set_run_name()
        input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
        preprocess_layer = tf_hub.KerasLayer(self.bert_preprocessor_url, name="preprocess")
        preprocessed_inputs = preprocess_layer(input_layer)
        encoder = tf_hub.KerasLayer(self.bert_model_url, trainable=True, name="BERT_encoder")
        outputs = encoder(preprocessed_inputs)
        net = outputs["pooled_output"]
        net = tf.keras.layers.Dropout(self.hparams[self.hyperparameters["dropout"]])(net)
        net = tf.keras.layers.Dense(2, activation=None, name="classifier")(net)
        self.model = tf.keras.Model(input_layer, net)

        self.model.compile(optimizer=self.optimizer,
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=["accuracy"])
        print(self.model.summary())
