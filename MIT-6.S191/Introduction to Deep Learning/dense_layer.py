import tensorflow as tf

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()

        # Initialize weights e bias
        self.W = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1, output_dim])

    def call(self, inputs):
        # Foward propagate the inputs, using matrix multiplication
        z = tf.matmul(inputs, self.W) + self.b

        # Feed through a non-linear activation, sigmoid
        output = tf.math.sigmoid(z)

        return output
