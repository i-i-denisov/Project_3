import tensorflow as tf

import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras.layers import Flatten
import pickle

#Hyperparamters
rate = 0.01
EPOCHS =30
BATCH_SIZE = 2000
dropout_rate=0.5
# Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
mu = 0
sigma = 0.1
mu_conv = 0
sigma_conv = 0.1
conv1_kernel=5
conv1_filter_depth=1
conv1_depth=12
conv2_kernel=5
conv2_filter_depth=12
conv2_depth=36
#TODO this is a wrong formula - to redefine it
FC_inputs_count=conv2_kernel*conv2_kernel*conv2_depth
fc_2_output_num=120
fc_3_output_num=84
noise_mu=0
noise_sigma=0.1
noise_lambda=5
# Implement the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.

# ### Input
# The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since MNIST images are grayscale, C is 1 in this case.
#
# ### Architecture
# **Layer 1: Convolutional.** The output shape should be 28x28x6.
#
# **Activation.** Your choice of activation function.
#
# **Pooling.** The output shape should be 14x14x6.
#
# **Layer 2: Convolutional.** The output shape should be 10x10x16.
#
# **Activation.** Your choice of activation function.
#
# **Pooling.** The output shape should be 5x5x16.
#
# **Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using `tf.contrib.layers.flatten`, which is already imported for you.
#
# **Layer 3: Fully Connected.** This should have 120 outputs.
#
# **Activation.** Your choice of activation function.
#
# **Layer 4: Fully Connected.** This should have 84 outputs.
#
# **Activation.** Your choice of activation function.
#
# **Layer 5: Fully Connected (Logits).** This should have 10 outputs.
#
# ### Output
# Return the result of the 2nd fully connected layer.
def LeNet(x,label_count,dropout_rate=0.0):
    #convolutionalpart
    fc0=LeNet_convR(x,dropout_rate=dropout_rate)
    fc0=Flatten()(fc0)
    #dense part
    logits=LeNet_FC(fc0,FC_inputs_count,label_count,dropout_rate=dropout_rate)
    return logits



def LeNet_convR (x,dropout_rate=0.0):
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    convR1_W = tf.Variable(tf.truncated_normal(shape=(conv1_kernel, conv1_kernel, conv1_filter_depth, conv1_depth), mean=mu_conv, stddev=sigma_conv))
    convR1_b = tf.Variable(tf.zeros(conv1_depth))
    convR1 = tf.nn.conv2d(x, convR1_W, strides=[1, 1, 1, 1], padding='VALID') + convR1_b

    # SOLUTION: Activation.
    convR1 = tf.nn.relu(convR1)

    # SOLUTION: Pooling. Input = 28x28x12. Output = 14x14x12.
    convR1 = tf.nn.max_pool2d(convR1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='conv1R')
    #
    # print (conv1.name)

    #Layer 2: Convolutional. Output = 10x10x32.
    convR2_W = tf.Variable(tf.truncated_normal(shape=(conv2_kernel, conv2_kernel, conv2_filter_depth, conv2_depth), mean=mu_conv, stddev=sigma_conv))
    convR2_b = tf.Variable(tf.zeros(conv2_depth))
    convR2 = tf.nn.conv2d(convR1, convR2_W, strides=[1, 1, 1, 1], padding='VALID') + convR2_b

    # SOLUTION: Activation.
    convR2 = tf.nn.relu(convR2)

    # SOLUTION: Pooling. Input = 10x10x32. Output = 5x5x32.
    convR2 = tf.nn.max_pool2d(convR2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='conv2R')
    return convR2


def LeNet_FC(fc0, FC_inputs_count, label_count, dropout_rate=0.0):
    #fc1_W = tf.Variable(tf.truncated_normal(shape=(2400, 400), mean=mu, stddev=sigma))
    #fc1_b = tf.Variable(tf.zeros(400))
    #fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    #fc1 = tf.nn.relu(fc1)
    # fc1=tf.nn.dropout(fc1,rate=dropout_rate)
    #  Layer 4: Fully Connected. Input = 400. Output = 120.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(FC_inputs_count, fc_2_output_num), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(fc_2_output_num))
    fc2 = tf.matmul(fc0, fc2_W) + fc2_b
    # Activation.
    fc2 = tf.nn.relu(fc2)
    fc2=tf.nn.dropout(fc2,rate=dropout_rate)

    #Layer 5: Fully Connected. Input = 120. Output = 84.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(fc_2_output_num, fc_3_output_num), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(fc_3_output_num))
    fc3 = tf.matmul(fc2, fc3_W) + fc3_b
    # Activation.
    fc3 = tf.nn.relu(fc3)
    fc3=tf.nn.dropout(fc3,rate=dropout_rate)

    #Layer 6: Fully Connected. Input = 84. Output = 43.
    fc4_W = tf.Variable(tf.truncated_normal(shape=(fc_3_output_num, label_count), mean=mu, stddev=sigma))
    fc4_b = tf.Variable(tf.zeros(label_count))
    logits = tf.matmul(fc3, fc4_W) + fc4_b
    return logits


# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# Note: that to get access to tf_activation, the session should be interactive which can be achieved with the following commands.
# sess = tf.InteractiveSession()
# sess.as_default()

# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and    max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1, plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    image_input = np.expand_dims(image_input, axis=0)
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    sess=tf.get_default_session()
    x=sess.graph.get_tensor_by_name('placeholder_x:0')
    activation = tf_activation.eval(session=sess, feed_dict={x: image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15, 15))
    for featuremap in range(featuremaps):
        plt.subplot(6, 8, featuremap + 1)  # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap))  # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min,
                       vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", cmap="gray")
    #plt.draw()


def evaluate(x_data, y_data):
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = x_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy_operation=sess.graph.get_tensor_by_name('accuracy_op:0')
        x= sess.graph.get_tensor_by_name('placeholder_x:0')
        y = sess.graph.get_tensor_by_name('placeholder_y:0')
        dropout_rate=sess.graph.get_tensor_by_name('placeholder_dropout_rate:0')
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, dropout_rate:0.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def dataset_load():
    train_file = open("C:/Tools/Udacity/Project_3/traffic signs/train.p", "rb")
    valid_file = open("C:/Tools/Udacity/Project_3/traffic signs/valid.p", "rb")
    test_file = open("C:/Tools/Udacity/Project_3/traffic signs/test.p", "rb")
    train_set = pickle.load(train_file)
    valid_set = pickle.load(valid_file)
    test_set = pickle.load(test_file)
    X_train, y_train = train_set['features'], train_set['labels']
    X_validation, y_validation = valid_set['features'], valid_set['labels']
    X_test, y_test = test_set['features'], test_set['labels']

    assert (len(X_train) == len(y_train))
    assert (len(X_validation) == len(y_validation))
    assert (len(X_test) == len(y_test))

    print()
    print("Image Shape: {}".format(X_train[0].shape))
    print()
    print("Training Set:   {} samples".format(len(X_train)))
    print("Validation Set: {} samples".format(len(X_validation)))
    print("Test Set:       {} samples".format(len(X_test)))

    return X_train, y_train, X_validation, y_validation, X_test, y_test

def dataset_normalize(x):
    #very rough image normalisation
    x=(x/1-128)/128
    return x

def dataset_grayscale(x,weights=[0.3,0.3,0.3]):
    assert x.shape[-1]==len(weights)
    grayscale=np.zeros_like(x[0,:,:,0])
    for i in range(x.shape[-1]):
        grayscale=(weights[i] * x[:,:,:,i]) + grayscale
    return (np.expand_dims(grayscale,axis=3))

def dataset_visualise(x_train, y_train, x_valid, y_valid, x_test=[], y_test=[]):
    # View a sample from the dataset.
    index = random.randint(0, len(x_train))
    image = x_train[index].squeeze()
    fig = plt.figure(figsize=(2, 2))
    plt.imshow(image)
    fig.suptitle("Image label {}, image shape: {}, datatype: {}, min_value: {}, max_value: {}".format(y_train[index], image.shape, image.dtype, np.min(image), np.max(image)))
    ##Plot label counts across datasets
    unique_labels_train, label_count_train = np.unique(y_train, return_counts=True)
    unique_labels_valid, label_count_valid = np.unique(y_valid, return_counts=True)
    unique_labels_test, label_count_test = np.unique(y_test, return_counts=True)
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=False)
    axs[0].bar(unique_labels_train, label_count_train)
    axs[0].set_ylabel("Label count")
    axs[0].set_xlabel("Train dataset")
    axs[1].bar(unique_labels_valid, label_count_valid)
    axs[1].set_xlabel("Validation dataset")
    axs[2].bar(unique_labels_test, label_count_test)
    axs[2].set_xlabel("Test dataset")
    if len(label_count_train) != len(label_count_valid):
        print("Datasets might be unbalanced. Train dataset label count: {}, Valid dataset label count: {}".format(len(label_count_train), len(label_count_valid)))
        fig.suptitle("Label distribution across datasets. Warning! Datasets might be unbalanced")
    plt.show()

def rate_decay(epoch=0):
    rate=0.1*np.exp(-epoch//5)+0.00001
    return rate

def image_add_gauss_noise(x):
    noise = np.random.default_rng().normal(noise_mu, noise_sigma, (x.shape))
    x=x+noise
    return x

def image_add_poisson_noise(x):
    noise = np.random.default_rng().poisson(noise_lambda, (x.shape))-noise_lambda
    x=x+noise
    return  x

def dataset_augment_poisson_noise(x,y):
    noised_x=image_add_poisson_noise(x)
    augm_x=np.vstack((x,noised_x))
    augm_y=np.hstack((y,y))
    return augm_x,augm_y

def dataset_augment_rotate_shift_w_balanced_count(x,y):
    unique_labels, label_count = np.unique(y, return_counts=True)
    max_label_num=np.amax(label_count)
    max_label_index=unique_labels[np.argmax(label_count)]
    for i in len(unique_labels):
        label=unique_labels[i]
        count=label_count[i]
        if count*3<max_label_num:
            x,y=augment_rotate_shift(x,y)
        elif count*2<max_label_num:
            x,y=augment_rotate(x,y)




