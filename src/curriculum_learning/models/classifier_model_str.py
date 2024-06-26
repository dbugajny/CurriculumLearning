import tensorflow as tf
from curriculum_learning.models.blocks import ConvBlock, DenseBlock


class ClassifierModel(tf.keras.Model):
    def __init__(
        self,
        input_shape,
        output_shape: int,
        conv_block_filters: list[int],
        conv_block_kernel_sizes: list[int],
        conv_block_strides: list[int],
        conv_block_dropout_rates: list[float],
        dense_block_units: list[int],
        dense_block_dropout_rates: list[float],
    ) -> None:


        self.conv_blocks = []
        for filters, kernel_size, strides, dropout_rate in zip(
            conv_block_filters, conv_block_kernel_sizes, conv_block_strides, conv_block_dropout_rates
        ):
            conv_block = ConvBlock(filters=filters, kernel_size=kernel_size, strides=strides, dropout_rate=dropout_rate)
            self.conv_blocks.append(conv_block)

        self.dense_blocks = []
        for units, dropout_rate in zip(dense_block_units, dense_block_dropout_rates):
            dense_block = DenseBlock(units=units, dropout_rate=dropout_rate)
            self.dense_blocks.append(dense_block)

        self.output_layer = tf.keras.layers.Dense(output_shape, activation="softmax")

        inputs = tf.keras.layers.Input(shape=input_shape, name='image')

        outputs = self.get_call(inputs)
        super().__init__(inputs=inputs, outputs=outputs)


    def get_call(self, x: tf.Tensor, training: bool = False, mask=None) -> tf.Tensor:
        for block in self.conv_blocks:
            x = block(x, training=training)

        x = tf.keras.layers.Flatten()(x)

        for block in self.dense_blocks:
            x = block(x, training=training)

        x = self.output_layer(x)

        return x
