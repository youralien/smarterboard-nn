Notes
===============

###Dropout
* Useful for big, complex nets that overfit.  Must determine if the current net has high-variance (overfitting) or high-bias (underfitting) on the training data
* how to create a "mean net" with dropout [https://class.coursera.org/neuralnets-2012-001/lecture/119]
	* are weights being halved in foxhound?
* can't control dropout for Input layer.

###Experiments 11/29:

* QUESTION: Alec suspected that the 100% train and test accuracy in the artifically generated dataset was due to the net picking up key set points
* ANSWER?: Augmentations of the training data should make the net work harder to learn more generalizable representations.
* METHOD: Random Flipping (lr, up, lr + ud), Translation (-.125, .125 percent of the image), and Rotation (-10, 10 degrees) for each image example running through the net during training.
* DATA: The overfitting we were seeing before went away.  The first attempt we had very close train and test accuracies in the 75% range.  We ran it for 100+ epochs and got in the 95% plus train/test accuracy range.
* INSIGHTS: We can squeeze every bit of juice from our small dataset with the help of augmentation to mitigate overfitting.  We can also use artifical data + augmentation to add a richer set of data as well, without the fear of learning artifical/unauthentic representations.

###Experiments 11/28:
* Adding Artifically Generated Images to Training 
	* Increased Train Accuracy from 75% -> 95%
	* Increased Test Accuracy from 65% -> 75%
* Using Artifically Generated Images for Training and Testing resulted in 100% train/test accuracy!
* Double the n_epochs -> 98 -> 99% train accuracy
* dropout is not appearantly fixing the high-variance
