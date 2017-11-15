# **Traffic Sign Recognition** 

## Writeup Template


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Project files


The project page is on [Github](https://github.com/pvishal-carnd/Traffic-Sign-Classifier). This writeup can at [writeup.md](writeup.md) and the Jupyter notebook, that has all the code can be accessed at [Traffic_Sign_Classifier.ipynb](Traffic_Sign_Classifier.ipynb). 

### Data Set Summary & Exploration

The dataset exploration was done primarily with `numpy`. `pandas` was used to read in the `csv` file with sign descriptions. The training, validation and test set has already been provided. The following list shows a summary.   


* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

##### Visualization of the dataset.

Each image is a 32x32x3 RGB array with integer values in the range `[0 255]'. We plot a random images of a few classes to see how they look like. 

![Sample Images][doc-images/datavis.png]

Next, we plot the histogram of the number of images for each class.

![Data histograms][doc-images/data-histograms.png]

From the shapes of each of the three distributions,  the validation and the test datasets, at least at the outset do seem to be drawn from the same distribution. However, the dataset is unbalanced. Certain classes are significantly under-represented with respect to the number of available samples in the training set. We will keep an eye out for these classes when we analyze the prediction model performance at a later stage.  

### Pre-processing

As a first step of the preprocessing pipeline, the images were converted to grayscale. This step is not strictly required but we noticed small improvement in the overall validation set performance with this conversion.   In their paper, Pierre Sermanet and Yann LeCun also report a marginal improvement with grayscale conversions. Having said that, certain classes of sign did significantly better after grayscale conversion. We will address this issue in a later section when we analyze the model performance.

After grayscaling, the images were normalized. We did this by linearly mapping all the pixel values to lie between `0` and `1`. Apart from helping the learning by keeping the means small and the cost function well distributed along all axes, this normalization steps also helps make the algorithm invariant to brightness changes. The image histograms become flatter and this helps bring up faint features. However, this method fails to work under cases where parts of the image are overexposed (perhaps due to the sky in the background). Local adaptive histogram equalization, local normalization or applying a non-linear transformation like gamma correction can potentially help such cases. We, nevertheless, continued with our simple min-max normalization scheme and the results are as follows:   

![Preprocessing results][doc-images/preproc.png]


### Data augmentation

There are two reasons for considering data augmentation for this model:
1. The dataset is highly unbalances. In some cases, there are only a little more than 200 images per class. We could therefore augment those under-represented classes with generated data.
2. As we shall see later, our models mostly overfit the data. In such cases, there is almost always room for more variation in the training set. We bring about this variation by generating jittered data.

We can choose among a very large set of transformations to create an augmented dataset. We however, focus on transformations that are realistic for our case. For example, flipping will not help as it completely changes most of the traffic signs. Tranformations like warp will not occur in practice and neither will shear (expect probably as a rolling shutter artifact at high speeds). Brightness changes, while very possible, get normalized away in our pre-processing step. Any color-based transformations are lost in our conversion to grayscale. These transformations are unlikely to improve the field performance of our model.  

We therefore, limit ourselves to the following transformations.  
- Translation
- Rotation
- Constrast 
- Crops
- Gaussion blur
- Additive noise
- Perspective

We use a library `imgaug` for this task. This library allows us to easily create a random augmentation pipeline and apply it to large image sets. It is also easily available on `PyPi` and can be simply installed with 

```
pip install imgaug
```

The following code snippet sets up this pipeline for us. The code comments are self-explanatory. We don't translate a lot of images because convolutions are fairly robust to translations. 

```
augPipeline = iaa.Sequential([
	# Crop all images with random percentage from 0 to 22%
    iaa.Crop(percent=(0, 0.22)), # random crops
    
	# With a probability of 0.3, apply either a random Gaussian 
    # blur or add random additive noise. 
    iaa.Sometimes(0.3, 
    iaa.OneOf([
    iaa.GaussianBlur(sigma=(0, 1)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0 ,0.07*255))
    ])),
    
    # Strengthen or weaken the contrast in some images 
    iaa.Sometimes(0.25, iaa.ContrastNormalization((0.75, 1.5))),
    
    # Translate some images
    iaa.Sometimes(0.1, iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})),
    
	# With a high probability, apply a perspective transform
    iaa.Sometimes(0.97,
                  iaa.PerspectiveTransform(scale=(0, 0.1))
                 ),

	# Apply affine transformations to each image.
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        rotate=(-6, 6),
        shear=(-15, 15)
    )
], random_order=True) # Apply the above transformations in a random order
```
The results of this pipeline are shown for a single randomly selected image.

![Augmentation pipeline test][doc-images/augmented.png]

We apply this augmentation mostly only to the underrepresented classes to bring up each of their sample size to about 1500 images per class. We confirm this by looking at the histogram for the augmented dataset.


![Augmented histogram][doc-images/augmented-hist.png]

### Design and Test a Model Architecture



##### Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


##### Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

##### Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

##### Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

##### Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

##### Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


