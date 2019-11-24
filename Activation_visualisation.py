import functions
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
X_train, y_train, X_validation, y_validation=functions.load_dataset()
all_one = np.copy(X_train[1])
all_one[:, 6:, :] = np.max(X_train[1])
with tf.Session() as sess:
    restorer = tf.train.import_meta_graph('lenet.meta')
    restorer.restore(sess, tf.train.latest_checkpoint('.'))
    print('input_image')
    plt.figure(0,figsize=(1, 1))
    img = X_train[1]
    #plt.ion()
    plt.imshow(img.squeeze(), cmap='gray')
    #plt.pause(0.0001)
    #plt.draw()

    print('first layer response')
    layer1 = sess.graph.get_tensor_by_name('conv1:0')
    layer2 = sess.graph.get_tensor_by_name('conv2:0')
    x = sess.graph.get_tensor_by_name('placeholder_x:0')
    functions.outputFeatureMap(img, layer1, plt_num=1)
    functions.outputFeatureMap(img, layer2, plt_num=2)
    print("Max memory use", sess.run(tf.contrib.memory_stats.MaxBytesInUse()) // 1024, "kbytes")
plt.show()
