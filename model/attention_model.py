import keras.losses
import tensorflow as tf
from .base_model import BaseModel
from .utils import positional_encoding

# used the self-attention model in 14 Neural Networks Sanity Checks.ipynb

class AttentionModel(BaseModel):
    num_heads = None

    def __init__(self, vocabulary_size: int, embedding_dim: int, hidden_dim: int, maximum_position_encoding: int,
                 num_heads: int, num_classes: int, epochs: int, batch_pipeline, hparam_manager,
                 embeddings_initializer=tf.keras.initializers.GlorotUniform(), embeddings_regularizer=None):
        self.name = 'attentionmodel'
        self.batch_pipeline = batch_pipeline
        self.hparam_manager = hparam_manager
        self.epochs = epochs

        self.model = Classifier(
            vocabulary_size=vocabulary_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            maximum_position_encoding=maximum_position_encoding,
            num_heads=num_heads,
            num_classes=num_classes,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            dropout_rate=self.hparam_manager.dropout
        )
        train_dataset_len = tf.data.experimental.cardinality(batch_pipeline.train_dataset).numpy()
        optimizer = self._get_optimizer(train_dataset_len)
        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )


class Classifier(tf.keras.Model):

    def __init__(self, vocabulary_size: int,  embedding_dim: int, hidden_dim: int, maximum_position_encoding: int,
                 num_heads: int, num_classes: int, embeddings_initializer: tf.keras.initializers,
                 embeddings_regularizer: tf.keras.regularizers, dropout_rate: int):
        super().__init__()

        assert embedding_dim % num_heads == 0, "embedding_dim must be a multiple of num_heads"
        # if the embedding layer is a constant then we must not train the layer.
        self.should_train_embedding_layer = not isinstance(embeddings_initializer, tf.keras.initializers.Constant)
        self.embedding = EmbeddingLayer(embedding_dim, vocabulary_size, maximum_position_encoding,
                                        embeddings_initializer, embeddings_regularizer, dropout_rate)
        self.encoder_layer = EncoderLayer(embedding_dim, hidden_dim, num_heads, dropout_rate=dropout_rate)
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.classification_layer = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)

    def call(self, x, training, return_attention_scores=False):
        padding_mask = self.create_padding_mask(x)
        x = self.embedding(x, self.should_train_embedding_layer)
        x, scores = self.encoder_layer(x, training, padding_mask)
        x = self.pooling(x)
        x = self.classification_layer(x)

        return (x, scores) if return_attention_scores is True else x

    @staticmethod
    def create_padding_mask(seq):
        seq = 1 - tf.cast(tf.math.equal(seq, 0), tf.float32)
        # add extra dimensions for the attention layer
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim: int, vocabulary_size: int, maximum_position_encoding: int,
                 embeddings_initializer: tf.keras.initializers, embeddings_regularizer: tf.keras.regularizers,
                 dropout_rate: int):

        super(EmbeddingLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_dim,
                                                   embeddings_initializer=embeddings_initializer,
                                                   embeddings_regularizer=embeddings_regularizer,
                                                   trainable=not(isinstance(embeddings_initializer,
                                                                            tf.keras.initializers.Constant)))
        self.pos_encoding = positional_encoding(maximum_position_encoding, embedding_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training):
        sequence_len = tf.shape(x)[1]
        # retrieving the embeddings and adding positional encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))  # scaling
        x += self.pos_encoding[:, :sequence_len, :]
        x = self.dropout(x, training=training)

        return x  # (batch_size, input_seq_len, embedding_dim)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim, hidden_dim, num_heads, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        head_dim = int(model_dim / num_heads)

        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads, head_dim, dropout=dropout_rate)
        # feedforward
        self.feed_forward1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.feed_forward2 = tf.keras.layers.Dense(model_dim)
        # layer normalization
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, padding_mask):
        # multihead-self-attention
        attention_output, attention_scores = self.self_attention(x, x, attention_mask=padding_mask, training=training,
                                                                 return_attention_scores=True)
        out1 = self.layer_norm1(attention_output + x)  # layer normalization and residual connection

        # feedforward network
        feed_forward_output = self.feed_forward1(out1)
        feed_forward_output = self.feed_forward2(feed_forward_output)
        feed_forward_output = self.dropout(feed_forward_output, training=training)

        out2 = self.layer_norm2(feed_forward_output + out1)  # layer normalization and residual connection

        return out2, attention_scores
