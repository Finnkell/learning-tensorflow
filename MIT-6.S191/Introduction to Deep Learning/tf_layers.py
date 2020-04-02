'''
Already implemented layers in tensorflow
'''

import tensorflow as tf

# a single Dense layer with two outputs
layer = tf.keras.layers.Dense(units=2)

# a multi output perceptron with two Dense layers, one with n ouputs and the second with 2 outputs
model = tf.keras.layers.Sequential([
    tf.keras.layers.Dense(n),
    tf.keras.layers.Dense(2)
])

# in general to a Deep Neural Network we have n Dense Laysers with n outputs
model = tf.keras.layers.Sequential([
    tf.keras.layers.Dense(n1),
    tf.keras.layers.Dense(n2),
    ...
    tf.keras.layers.Dense(2)
])