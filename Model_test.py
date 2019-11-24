import tensorflow as tf
from functions import evaluate
from tensorflow.examples.tutorials.mnist import input_data
from datetime import  datetime

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)

X_test, y_test = mnist.test.images, mnist.test.labels
BATCH_SIZE = 2000

#correct_prediction = tf.math.equal(tf.math.argmax(logits, 1), tf.math.argmax(one_hot_y, 1))
#accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

tf.reset_default_graph()
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('lenet.meta')
    new_saver.restore(sess, 'lenet')
    graph = tf.get_default_graph()
    #accuracy_operation = graph.get_operation_by_name('accuracy_operation').outputs[0]
    file_writer = tf.summary.FileWriter(logdir='checkpoint_log_dir/faceboxes', graph=graph)

    #test_accuracy = evaluate(X_test, y_test)
    #print("Test Accuracy = {:.3f}".format(test_accuracy))
