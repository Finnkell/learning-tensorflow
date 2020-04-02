'''
Some of Gradient Descent Algorithms, with a fixed and adapted learning rate 
'''

import tensorflow as tf

'''
With a fixed learning rate
'''
# SGD
tf.keras.optimizers.SGD()

'''
With a adpated learning rate
'''
# Adam
tf.keras.optimizers.Adam()

# Adadelta
tf.keras.optimizers.Adadelta()

# Adagrad
tf.keras.optimizers.Adagrad()

# RMSProp
tf.keras.optimizers.RSMProp()