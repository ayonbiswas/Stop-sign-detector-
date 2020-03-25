import cv2
from skimage import data, util
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import numpy as np
import os
import time


def calculate_likelihood(x,mu,cov,prior):
	#calculates log likelihood of a pixel x for a given gaussian distribution
    term1 = x - mu
    t = 0.5*(term1)@np.linalg.inv(cov)
    term2 = np.multiply(0.5*(term1)@np.linalg.inv(cov),term1).sum(axis = 1)

    return np.log(prior) - 0.5*np.log(np.linalg.det(cov)) - term2

def predict(x):
	#predicts class of a pixel x
    y_pred = calculate_likelihood(x,mu1,cov1,prior1) > calculate_likelihood(x,mu0,cov0,prior0) 
    return y_pred.reshape(-1)
    
def create_mask(img):
	#Compute the segmentation mask for an image
    w, h ,c = img.shape
    mask = np.zeros((w,h))
    for i in range(w):
        mask[i,:] = predict(img[i,:])

    return mask

#normalise pixels to [0,1]
mat_0 = np.load("./class_0_rgb.npy")/255.
mat_1 = np.load("./class_1_rgb.npy")/255.

#perform train-test split
train_1 = mat_1[:round(0.95*mat_1.shape[0])]
test_1 = mat_1[round(0.95*mat_1.shape[0]):]
train_0 = mat_0[:round(0.95*mat_0.shape[0])]
test_0 = mat_0[round(0.95*mat_0.shape[0]):]

train_X = np.concatenate((train_1,train_0),axis=0)
train_Y = np.concatenate((np.ones((train_1.shape[0])),np.zeros((train_0.shape[0]))))

test_X = np.concatenate((test_1,test_0),axis=0)
test_Y = np.concatenate((np.ones((test_1.shape[0])),np.zeros((test_0.shape[0]))))

#calculate mean
mu1 = np.mean(train_1,axis=0)
mu0 = np.mean(train_0, axis = 0)

#calculate covariance 
cov1 = np.dot((train_1 - mu1).T,(train_1-mu1))/train_1.shape[0]
cov0 = np.dot((train_0 - mu0).T,(train_0-mu0))/train_0.shape[0]

#calculate class priors
prior1 = train_1.shape[0]/(train_1.shape[0]+train_0.shape[0])
prior0 = train_0.shape[0]/(train_1.shape[0]+train_0.shape[0])

res = predict(test_X)
print("test_accuracy:", np.mean(res*1== test_Y.reshape(-1)))


# np.save("./cov1.npy",cov1)
# np.save("./cov0.npy",cov0)
# np.save("./mu1.npy",mu1)
# np.save("./mu0.npy",mu0)
# np.save("./prior1.npy",prior1)
# np.save("./prior0.npy",prior0)