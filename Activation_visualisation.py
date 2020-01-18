import functions
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
img_num=1712
x_train, y_train, X_validation, y_validation, __, __=functions.dataset_load()
#x_train=functions.dataset_grayscale(x_train)
with tf.Session() as sess:
    restorer = tf.train.import_meta_graph('lenet.meta')
    restorer.restore(sess, tf.train.latest_checkpoint('.'))
    print('input_image')
    fig=plt.figure(0,figsize=(1, 1))
    img = x_train[img_num]
    #plt.ion()
    plt.imshow(img.squeeze(), cmap='gray')
    fig.suptitle("Image label {}".format(y_train[img_num]))
    #plt.show()
    #plt.pause(0.0001)
    #plt.draw()

    print('first layer response')
    layer1R = sess.graph.get_tensor_by_name('conv1R:0')
    layer2R = sess.graph.get_tensor_by_name('conv2R:0')
    #layer1G = sess.graph.get_tensor_by_name('conv1G:0')
    #layer2G = sess.graph.get_tensor_by_name('conv2G:0')
    #layer1B = sess.graph.get_tensor_by_name('conv1B:0')
    #layer2B = sess.graph.get_tensor_by_name('conv2B:0')
    x = sess.graph.get_tensor_by_name('placeholder_x:0')
    functions.outputFeatureMap(img, layer1R, plt_num=1)
    functions.outputFeatureMap(img, layer2R, plt_num=2)
    #functions.outputFeatureMap(img, layer1G, plt_num=3)
    #functions.outputFeatureMap(img, layer2G, plt_num=4)
    #functions.outputFeatureMap(img, layer1B, plt_num=5)
    #functions.outputFeatureMap(img, layer2B, plt_num=6)
    print("Max memory use", sess.run(tf.contrib.memory_stats.MaxBytesInUse()) // 1024, "kbytes")
plt.show()
