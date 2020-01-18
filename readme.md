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
Certain number of steps of this project can be distinguished:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./writeup/random_sign_visualisation.png "Random sign"
[image2]: ./writeup/Dataset_label_count.png "Dataset sign distribution"
[image3]: ./writeup/Dataset_examples.png "Dataset overwiev"
[image4]: ./writeup/warp.png "Warp Example"
[image5]: ./writeup/warped_lines_drawn.jpg "Fit Visual"


##Dataset exploration

This dataset consisnts of 51.839 32x32 RGB images distributed across 43 classes.
Each image contains only one traffic sign.
Examples of images for each sign in dataset:
![Alt_text][image3]

One image at closeup: 
![Alt text][image1]  

I split into three sets for training, validation and testing:
- Training Set:   34799 samples
- Validation Set: 4410 samples
- Test Set:       12630 samples

This is how images are distributed across this sets:
![Alt_text][image2]
  


