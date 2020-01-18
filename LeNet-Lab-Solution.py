#!/usr/bin/env python
# coding: utf-8

# # LeNet Lab Solution
# Source: Yan LeCun


import random
import matplotlib.pyplot as plt


import numpy as np
from sklearn.utils import shuffle

import tensorflow as tf
import functions


x_train, y_train, x_valid, y_valid, x_test, y_test=functions.dataset_load()
unique_labels_train, label_dist_train=np.unique(y_train, return_counts=True)
unique_labels_valid, label_dist_valid=np.unique(y_valid, return_counts=True)

# ## Preprocess Data
x_train, y_train = shuffle(x_train, y_train)
functions.dataset_visualise(x_train,y_train,x_valid,y_valid,x_train,y_train)
x_train= functions.dataset_normalize(x_train)
x_valid=functions.dataset_normalize(x_valid)
(img_width,img_height,img_layers)=x_train[0].shape
print (x_train[0].shape)
# ## Features and Labels
# Train LeNet to classify German traffic signs dataset
# `x` is a placeholder for a batch of input images.
# `y` is a placeholder for a batch of output labels.
label_num=len(label_dist_train)
x = tf.placeholder(tf.float32,  (None, img_width,img_height,img_layers), name='placeholder_x')
y = tf.placeholder(tf.int32, (None), name='placeholder_y')
dropout_rate = tf.placeholder(tf.float32, name='placeholder_dropout_rate')
one_hot_y = tf.one_hot(y, label_num)
# ## Training Pipeline
# Create a training pipeline that uses the model to classify MNIST data.
logits = functions.LeNet(x,label_num)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=functions.rate)
training_operation = optimizer.minimize(loss_operation)

# ## Model Evaluation
# Evaluate how well the loss and accuracy of the model for a given dataset.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')

# ## Train the Model
# Run the training data through the training pipeline to train the model.
# Before each epoch, shuffle the training set.
# After each epoch, measure the loss and accuracy of the validation set.
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)

    print("Training...")
    print()
    for epoch in range(functions.EPOCHS):
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, num_examples, functions.BATCH_SIZE):
            end = offset + functions.BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, dropout_rate:functions.dropout_rate})

        validation_accuracy = functions.evaluate(x_valid, y_valid)
        print("EPOCH {} ...".format(epoch + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Max memory use", sess.run(tf.contrib.memory_stats.MaxBytesInUse()) // 1024, " Kbytes")
    print("Model saved")







