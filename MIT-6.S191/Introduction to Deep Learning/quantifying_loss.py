'''
Two examples to Quantifying Loss in Neural Network
'''

import tensorflow as tf

# Binary Cross Entropy Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, predictions))

# Mean Squared Error Loss
loss = tf.reduce_mean(tf.square(tf.subtract(y, predictions)))