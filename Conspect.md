#Momentum
Nesterov momentum
Principle component analysis - eigen samples
Autoencoding





#Padding
_**Valid**_ padding is not adding any extra pixels, just following the dimensions of image
_**Same**_ padding is adding zeroes outisde the image to match output dimensions with input dimensions
#Stride
Stride is a number of pixels we shift filter every step

#Dimensions of convolutional networks
##Given:

our input layer has a width of W and a height of H
our convolutional layer has a filter size F
we have a stride of S
a padding of P
and the number of filters K,
the following formula gives us the width of the next layer: `W_out =[ (W−F+2P)/S] + 1`.

The output height would be H_out = `[(H-F+2P)/S] + 1`.

And the output depth would be equal to the number of filters `D_out = K`.

The output volume would be `W_out * H_out * D_out`.

#Conv networks in tensorflow
```input = tf.placeholder(tf.float32, (None, 32, 32, 3))
filter_weights = tf.Variable(tf.truncated_normal((8, 8, 3, 20))) # (height, width, input_depth, output_depth)
filter_bias = tf.Variable(tf.zeros(20))
strides = [1, 2, 2, 1] # (batch, height, width, depth)
padding = 'SAME'
conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias
```

**_SAME_** Padding, the output height and width are computed as:

`out_height = ceil(float(in_height) / float(strides[1]))`

`out_width = ceil(float(in_width) / float(strides[2]))`

**_VALID_** Padding, the output height and width are computed as:

`out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))`

`out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))`


