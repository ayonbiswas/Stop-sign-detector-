# ECE276A Project 1 : Color Segmentation
Stop sign detection using pixel classification and shape detection



Description
===========
This is project on Color Segmentation is done by Ayon Biswas[A53318795]. We tried Gaussian discriminant analysis and logistic regression. We decided finally decided to use Gaussian discriminant analysis.

Code organization
=================
*stop_sign_detector.py -- Run the script for creating mask and detect stop signs. Use segment_image(img) to obtain a segmented image using the trained color classifier. Use get_bounding_box(img) to get the bounding box of the stop sign in an image.

*label.py -- Run the script to hand label pixels to get pixel coordinates and produce a mask for each image for different classes. The masks are stored as numpy files in two separate folders for red and non-red classes.
 
*extract_pixels.py -- Use the script to use the mask of pixel coordinates and extract pixel vectors from images and save as a numpy array.

*GDA.py -- Run The script to train a Gaussian Discriminant model and save the mean, covariance and priors as numpy array.

*logistic_reg.py -- Run The script to train a logistic regression model and save the weight matrix. 

*.npy files -- prior, covariance, mean of trained gaussian discriminant model for two classes(0 - nonred and 1 - red)

Datasets
========
The dataset was provided with the starter code.

Models
======
The pretrained weights for GDA: for class 1(red) - prior1.npy, mu1.npy, cov1.npy and class 0(non-red) - prior0.npy, mu0.npy, cov0.npy

Acknowledgements
================
We thank Prof. Nikolay Atanasov and TAs for their guidance and support.


