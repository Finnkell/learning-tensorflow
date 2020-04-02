'''
Put all the concepts together
'''

import tensorflow as tf

# create a model
model = tf.keras.layers.Sequential([...])

# pick a optimizer
optimizer = tf.keras.optimizer.Adam()

while True: 

    # forward pass through the network
    prediction = model(x)

    with tf.GradientTape() as tape:
        # compute the loss
        loss = compute_loss(y, prediction)

    # update the weights using the gradient
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))