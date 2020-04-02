'''
Base algorithm to Gradient Descent
'''

import tensorflow as tf

# start the random values for weights
weights = tf.Variable([tf.random.nomal()])

while True:
    with tf.GradientTape() as g:
        loss = compute_loss(weight)
        gradient = g.gradient(loss, weights)

    # update the weights W - n*(Dj(W)/DW), where n = learning rate (lr)
    weights = weights - lr * gradient

