import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int, strides: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor, training: bool = False, mask=None) -> tf.Tensor:
        x = self.conv(inputs)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.dropout(x, training=training)

        return x


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, units: int, dropout_rate: float) -> None:
        super().__init__()
        self.dense = tf.keras.layers.Dense(units)
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor, training: bool = False, mask=None) -> tf.Tensor:
        x = self.dense(inputs)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.dropout(x, training=training)

        return x
