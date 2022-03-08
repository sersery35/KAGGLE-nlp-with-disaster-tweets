import keras
import tensorflow as tf
import tensorflow_text
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

    def __init__(self, batch_pipeline, hparam_manager, num_classes=2, epochs=10,
                 bert_model_url="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1",
                 bert_preprocessor_url="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"):
        self.name = 'simplebertmodel'
        self.hparam_manager = hparam_manager
        self.batch_pipeline = batch_pipeline
        self.epochs = epochs
        train_dataset_len = tf.data.experimental.cardinality(batch_pipeline.train_dataset).numpy()
        optimizer = self._get_optimizer(train_dataset_len)
        self._set_model(bert_preprocessor_url, bert_model_url, optimizer, num_classes)

    def _set_model(self, bert_preprocessor_url: str, bert_model_url: str, optimizer: tf.keras.optimizers,
                   num_classes: int):
        input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
        preprocess_layer = tf_hub.KerasLayer(bert_preprocessor_url, name="preprocess")
        preprocessed_inputs = preprocess_layer(input_layer)
        encoder = tf_hub.KerasLayer(bert_model_url, trainable=True, name="BERT_encoder")
        outputs = encoder(preprocessed_inputs)
        pooled_outputs = outputs["pooled_output"]
        dropout_outputs = tf.keras.layers.Dropout(self.hparam_manager.dropout)(pooled_outputs)
        outputs = tf.keras.layers.Dense(num_classes, activation=tf.nn.sigmoid, name="classifier")(dropout_outputs)

        self.model = tf.keras.Model(input_layer, outputs)

        self.model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=["accuracy"])
        print(self.model.summary())
