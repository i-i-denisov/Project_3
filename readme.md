The first problem involves normalizing the features for your training and test data.

Implement Min-Max scaling in the `normalize()` function to a range of `a=0.1` and `b=0.9`. After scaling, the values of the pixels in the input data should range from 0.1 to 0.9.

Since the raw notMNIST image data is in [grayscale](https://en.wikipedia.org/wiki/Grayscale), the current values range from a min of 0 to a max of 255.

Min-Max Scaling:
$
X'=a+{\frac {\left(X-X_{\min }\right)\left(b-a\right)}{X_{\max }-X_{\min }}}
$

*If you're having trouble solving problem 1, you can view the solution [here](https://github.com/udacity/CarND-TensorFlow-Lab/blob/master/solutions.ipynb).*

#Traffic sign classifier
---
Goal of this project is to train a convolutional neural network to classify traffic sign images using german traffic sings [dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

[//]: # (Image References)

[image1]: ./Writeup/random sign visualisation.png "Random sign"
[image2]: ./Writeup/Dataset label count.png "Dataset sign distribution"
[image3]: ./Writeup/combined_binary.jpg "Binary Example"
[image4]: ./Writeup/warp.png "Warp Example"
[image5]: ./Writeup/warped_lines_drawn.jpg "Fit Visual"


###Dataset exploration

This dataset consisnts of 51 839 labeled 32x32 RGB images, which we split into three sets for training, validation and testing:

- Training Set:   34799 samples
- Validation Set: 4410 samples
- Test Set:       12630 samples

