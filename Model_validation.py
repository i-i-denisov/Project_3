import functions
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
with tf.Session() as sess:
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    X_test, y_test = mnist.test.images, mnist.test.labels
    X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    restorer = tf.train.import_meta_graph('lenet.meta')
    restorer.restore(sess, './lenet')
    accuracy_operation = sess.graph.get_tensor_by_name('accuracy_op:0')
    x = sess.graph.get_tensor_by_name('placeholder_x:0')
    y = sess.graph.get_tensor_by_name('placeholder_y:0')
    test_accuracy = functions.evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    print("Max memory use", sess.run(tf.contrib.memory_stats.MaxBytesInUse()) // 1024, " Kbytes")