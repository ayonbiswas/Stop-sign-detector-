import cv2
from skimage import data, util
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import numpy as np
import os
import time

def evaluate(W,X,Y):
    X = np.hstack((X,np.ones((X.shape[0],1))))
    y_pred = np.dot(X,W)
    y_pred[y_pred > 0] = 1
    y_pred[y_pred < 0] = -1
    accuracy = np.sum(y_pred == Y)/Y.shape[0]
    print(accuracy)

def predict(W,X):
    X = np.hstack((X,np.ones((X.shape[0],1))))
#     print(X.shape)
    y_pred = np.dot(X,W)
    y_pred[y_pred > 0] = 1.
    y_pred[y_pred < 0] = -1.
#     print(y_pred.shape)
    return y_pred.reshape(-1)


def logistic_regression(X,Y,num_steps, lr):
    no_samples= X.shape[0]
    X = np.hstack((X,np.ones((no_samples,1))))
    W = np.zeros((X.shape[1],1))
    cntr = 0
    loss_prev = -2
    loss_cur = -1
    iter_ = 0
    delta_loss =-1
    while( iter_ < num_steps):
        Y_X_prod = Y*X
        output = sigmoid(np.dot(Y_X_prod,W))
        loss_cur = -np.mean(np.log(output))
        gradient = (np.dot(Y_X_prod.T,(1-output)))/no_samples
        W = W + lr*gradient
        print("iter:", iter_,"loss: ",loss_cur)
        iter_ += 1
        
    return W

mat_0 = np.load("./class_0_rgb.npy")/255.
mat_1 = np.load("./class_1_rgb.npy")/255.

train_1 = mat_1[:round(0.95*mat_1.shape[0])]
test_1 = mat_1[round(0.95*mat_1.shape[0]):]
train_0 = mat_0[:round(0.95*mat_0.shape[0])]
test_0 = mat_0[round(0.95*mat_0.shape[0]):]

train_X = np.concatenate((train_1,train_0),axis=0)
train_Y = np.concatenate((np.ones((train_1.shape[0])),np.zeros((train_0.shape[0]))))

test_X = np.concatenate((test_1,test_0),axis=0)
test_Y = np.concatenate((np.ones((test_1.shape[0])),np.zeros((test_0.shape[0]))))

W = logistic_regression(train_X,train_Y,3000, 1e-3)

np.save("./W_log_reg.npy",W)