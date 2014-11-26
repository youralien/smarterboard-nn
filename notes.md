Notes
===============

###Dropout
* Useful for big, complex nets that overfit.  Must determine if the current net has high-variance (overfitting) or high-bias (underfitting) on the training data
* how to create a "mean net" with dropout [https://class.coursera.org/neuralnets-2012-001/lecture/119]
	* are weights being halved in foxhound?
* can't control dropout for Input layer.


###Experiments:
* Adding Artifically Generated Images to Training 
	* Increased Train Accuracy from 75% -> 95%
	* Increased Test Accuracy from 65% -> 75%
* Using Artifically Generated Images for Training and Testing resulted in 100% train/test accuracy!
* Double the n_epochs -> 98 -> 99% train accuracy
* dropout is not appearantly fixing the high-variance
