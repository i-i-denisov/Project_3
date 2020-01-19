import functions
import tensorflow as tf
import numpy as np
with tf.Session() as sess:
    x_train, y_train, x_valid, y_valid, x_test, y_test = functions.dataset_load()
    #x_test = functions.dataset_grayscale(x_test)
    restorer = tf.train.import_meta_graph('lenet.meta')
    restorer.restore(sess, './lenet')
    accuracy_operation = sess.graph.get_tensor_by_name('accuracy_op:0')
    x = sess.graph.get_tensor_by_name('placeholder_x:0')
    y = sess.graph.get_tensor_by_name('placeholder_y:0')
    test_accuracy = functions.evaluate(x_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    print("Max memory use", sess.run(tf.contrib.memory_stats.MaxBytesInUse()) // 1024, " Kbytes")