#!/usr/bin/env python
# coding: utf-8

# # LeNet Lab Solution
# Source: Yan LeCun

# ## Load Data
#
# Load the MNIST data, which comes pre-loaded with TensorFlow.
#
# You do not need to modify this section.

import random
import matplotlib.pyplot as plt


import numpy as np
from sklearn.utils import shuffle

import tensorflow as tf
import functions

# In[2]:

X_train, y_train, X_validation, y_validation=functions.load_dataset()


# ## Visualize Data
#
# View a sample from the dataset.
#
# You do not need to modify this section.

# In[4]:


index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

fig=plt.figure(figsize=(2, 2))
plt.imshow(image, cmap="gray")
fig.suptitle("Label"+str(y_train[index])+' image shape'+ str(image.shape)+' dtype' +str(image.dtype))
plt.show()
# ## Preprocess Data
#
# Shuffle the training data.
#
# You do not need to modify this section.

# In[5]:


X_train, y_train = shuffle(X_train, y_train)





# ## Features and Labels
# Train LeNet to classify [MNIST](http://yann.lecun.com/exdb/mnist/) data.
#
# `x` is a placeholder for a batch of input images.
# `y` is a placeholder for a batch of output labels.
#
# You do not need to modify this section.


x = tf.placeholder(tf.float32, (None, 32, 32, 1), name='placeholder_x')
y = tf.placeholder(tf.int32, (None), name='placeholder_y')
one_hot_y = tf.one_hot(y, 10)

# ## Training Pipeline
# Create a training pipeline that uses the model to classify MNIST data.
#
# You do not need to modify this section.




logits, layer1, layer2 = functions.LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=functions.rate)
training_operation = optimizer.minimize(loss_operation)

# ## Model Evaluation
# Evaluate how well the loss and accuracy of the model for a given dataset.
#
# You do not need to modify this section.




correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')








# ## Train the Model
# Run the training data through the training pipeline to train the model.
#
# Before each epoch, shuffle the training set.
#
# After each epoch, measure the loss and accuracy of the validation set.
#
# Save the model after training.
#
# You do not need to modify this section.




saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(functions.EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, functions.BATCH_SIZE):
            end = offset + functions.BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = functions.evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")







